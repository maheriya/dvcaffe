#! /usr/bin/env python
#
#   File = mergeLists.py
#   Opens all images from a directory one after another. Allows user
#   to work on one image, save before opening another image.
#
############################################################################
#
import sys, os
from time import sleep
import pickle
import pygtk
from random import randrange

pygtk.require("2.0")
import gtk
import glob
import random
import re

srcDir = os.path.join(os.environ['HOME'], "Projects/IMAGES/dvia/small/png")

scriptpath = os.path.dirname(os.path.realpath( __file__ ))

SIG_OK     = -5
SIG_CANCEL = -6
SIG_YES    = -8 
SIG_NO     = -9

def questionBox(msg):
    btype=gtk.MESSAGE_QUESTION
    flag = gtk.DIALOG_DESTROY_WITH_PARENT
    msgBox = gtk.MessageDialog(None, flag, btype, gtk.BUTTONS_YES_NO, msg)
    resp = msgBox.run()
    msgBox.destroy()
    return resp

def msgBox(msg,btype=gtk.MESSAGE_INFO):
    flag = gtk.DIALOG_MODAL|gtk.DIALOG_DESTROY_WITH_PARENT
    msgBox = gtk.MessageDialog(None, flag, btype, gtk.BUTTONS_OK, msg)
    resp = msgBox.run()
    msgBox.destroy()

class DialogBox:
    def __init__(self, srcdir, filelist):
        self.srcdir    = srcdir
        self.filelist  = filelist
        self.img       = None
        self.disp      = None

        self.gladefile = os.path.join(scriptpath, "openImages.glade") 
        self.wtree = gtk.Builder()
        self.wtree.add_from_file(self.gladefile)
        funcmap = {
                "on_next_button_clicked"     : self.next,
                "on_quit_button_clicked"     : self.quit,
                "on_mainWindow_destroy"      : self.quit,
        }
        self.wtree.connect_signals(funcmap)

        ## Get all the handles
        self.win = self.wtree.get_object("dialogWindow")
        self.win.show_all()
        
        self.srcfiles = []
        for fname in filelist:
            if not fname.lower().endswith('.xcf'):
                continue # skip non-xcf files
            self.srcfiles.append(fname)
        # Find all of the xcf files in the list
        if len(self.srcfiles) == 0:
            msgBox("Source directory didn't contain any XCF images.", gtk.MESSAGE_ERROR)
            return

        # Open the first image from the list and then let self.next take over based on user input
        self.next(None)
        gtk.main()


    def quit(self, widget):
        self.saveImage()
        try:self.win.destroy()
        except: pass
        gtk.main_quit()

    def next(self, widget):
        if len(self.srcfiles)==0:
            msgBox("No more files list to edit. We are done!", gtk.MESSAGE_INFO)
            self.quit()
        if len(gimp.image_list()) > 0 and self.img is not None:
            self.saveImage()

        srcfile = os.path.join(self.srcdir, self.srcfiles[0])
        self.srcfiles = self.srcfiles[1:] # pop_front() 
        print 'Opening ' + srcfile
        self.img = pdb.gimp_file_load(srcfile, srcfile)
        self.disp = pdb.gimp_display_new(self.img)

    def saveImage(self):
        if len(gimp.image_list()) == 0 or self.img is None:
            return
        pdb.gimp_xcf_save(0, self.img, self.img.active_drawable, self.img.filename, self.img.filename)
        pdb.gimp_image_clean_all(self.img)
        pdb.gimp_display_delete(self.disp)


# Shiva
def openLabels(srcDir):
    ###
    filelist = glob.glob(srcDir+'/*/labels_NP.txt')
    # print filelist

    vlabels = []
    tlabels = []
    print filelist

    for filename in filelist:
        dname = os.path.dirname(filename).split('/')[-1]
        print dname
        filePtr = open(filename, 'r')
        print filePtr
        labels = []
        print ("Starting with length of labels = {}".format(len(labels)))
        for line in filePtr.readlines():
            labels.append(os.path.join(dname,line.strip()))
                          
        random.shuffle(labels)
        nval = int(len(labels)*0.1)
        print ("Found {} labels".format(len(labels)))
        print ("validation labels to make {}".format(nval))
        cval = 0
        while(cval<nval):   
            random_index = randrange(0, (len(labels)-1))
            print labels[random_index]
            print labels[random_index].split('_')
            search_reg = labels[random_index].split('_')
            sr = '_'.join(search_reg[0:4])
            print sr

            for l in labels:
                m = re.search(sr, l)
                if m:
                    cval = cval + 1
                    vlabels.append(l)
                    labels.remove(l)
        for l in labels:
            tlabels.append(l)
                
        print len(vlabels)
        print len(tlabels)            
            
    print len(vlabels)
    print len(tlabels)
    random.shuffle(tlabels)
    random.shuffle(vlabels)
    outFile = open('train_NP.txt', 'w')
    outFile.write('\n'.join(tlabels))
    outFile.close()
    outFile = open('val_NP.txt', 'w')
    outFile.write('\n'.join(vlabels))
    outFile.close()
    
def openLabelsSimple(srcDir):
    ###
    filelist = glob.glob(srcDir+'/*/*.txt')
    # print filelist
    labels = []
    for filename in filelist:
        dname = os.path.dirname(filename).split('/')[-1]
        print dname
        filePtr = open(filename, 'r')
        for line in filePtr.readlines():
            labels.append(os.path.join(dname,line.strip()))
                          
    random.shuffle(labels)
    ntrain = int(len(labels)*0.9)
    nval = len(labels) - ntrain
    tlabels = labels[0:ntrain-1]
    vlabels = labels[ntrain:]
    outFile = open('train.txt', 'w')
    outFile.write('\n'.join(tlabels))
    outFile.close()
    outFile = open('val.txt', 'w')
    outFile.write('\n'.join(vlabels))
    outFile.close()
    
    
openLabels(srcDir)
#openLabelsSimple(srcDir)
