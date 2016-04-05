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

pygtk.require("2.0")
import gtk
import glob
import random

srcDir = os.path.join(os.environ['HOME'], "Projects/IMAGES/dvia/png")

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
