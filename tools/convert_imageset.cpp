// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::tuple<std::string, int, float, float> > lines;
  std::string filename;
  int label;
  float npx, npy;
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "Color? " << is_color << std::endl;;
  std::cout << "Encoded? " << encoded << std::endl;;
  std::cout << "Encode type: " << encode_type << std::endl;;
  std::cout << "Shuffle? " << FLAGS_shuffle << std::endl;;
  std::cout << "Input list file: " << argv[2] << std::endl;;
  std::cout << "------------------------------------------" << std::endl;

  std::cout << "Processing input list file: " << argv[2] << std::endl;;
  while (infile >> filename >> label >> npx >> npy) {
    //std::cout << "\tImage file: " << filename << "\tLabel: " << label << std::endl;
    lines.push_back(std::make_tuple(filename, label, npx, npy));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = std::get<0>(lines[line_id]);
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + std::get<0>(lines[line_id]),
        std::get<1>(lines[line_id]), resize_height, resize_width, is_color,
        enc, &datum);
    datum.set_npx(std::get<2>(lines[line_id]));
    datum.set_npy(std::get<3>(lines[line_id]));
    if (status == false) {
        std::cerr << "Failed to read image " << std::get<0>(lines[line_id]) << std::endl;
        continue;
    }
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        std::cout << "---\nInitial image parameters  datum.channels(): " << datum.channels()
                  << "\ndatum.height(): " << datum.height()
                  << "\ndatum.width(): " << datum.width() << "\n------" << std::endl;
        std::cout << "Initial image size " << datum.channels() * datum.height() * datum.width() << std::endl;
        const std::string& data = datum.data();
        std::cout << "Image size (datum.data.size()): " << data.size() << std::endl;
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        std::cout << "Image size (datum.data.size()): " << data.size() << std::endl;
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size " << data.size();
      }
    }
    // sequential
    //string key_str = caffe::format_int(line_id, 8) + "_" + std::get<0>(lines[line_id]);
    string key_str = caffe::format_int(line_id, 8); // + "_" + std::get<0>(lines[line_id]);
    std::cout << "Key: " << key_str << "\tFile: " << std::get<0>(lines[line_id]) << std::endl;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
