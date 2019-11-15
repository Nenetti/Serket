# -*- coding: utf-8 -*-
from __future__ import print_function,unicode_literals, absolute_import
import os
import sys
import numpy as np
import random


sys.path.append( "../" )

from . import julius
from . import lang_model
from . import utils
from . import histogram
import serket as srk


def mkdir( path ):
    if not os.path.exists(path):
        os.mkdir( path )

class WordRecog(srk.Module):
    def __init__(self, wave_files, nbest=10, threshold=1, lmp=[8.0, -2.0], name="speech_recog"):
        super(WordRecog, self).__init__(name, True)
        mkdir( self.get_name() )
        self.__nit = 0
        self.__nbest = nbest
        self.__threshold = threshold
        self.__wave_files = wave_files
        self.__julius = julius.Julius( os.path.join( self.get_name(), "julius" ), lmp )


    def initialize(self):
        fname_recogres = os.path.join( self.get_name(), "000", "recogres.txt" )
        fname_selected_sen = os.path.join( self.get_name(), "000", "selected_sentences.txt" )
        fname_lm = os.path.join( self.get_name(), "000", "lm" )
        fname_wordhist = os.path.join( self.get_name(), "000", "word" )

        self.__wave_id = []
        sentences = []
        select_sentences = []
        for d in range(len(self.__wave_files)):
            for n in range(len(self.__wave_files[d])):
                recogres = self.__julius.recog_kana( self.__wave_files[d][n], self.__nbest )
                select_sentences.append( recogres[0].replace("ー", "") )
                for i in range(self.__nbest):
                    self.__wave_id.append((d,n,i))
                    if i < len(recogres):
                        sentences.append( recogres[i].replace("ー", "") )
                    else:
                        sentences.append("")

        utils.save_lines( sentences, fname_recogres )
        utils.save_lines( select_sentences, fname_selected_sen )

        lang_model.makelm( fname_selected_sen, fname_lm )
        self.__obj_hist, self.__sen_hist = histogram.make_histogram( self.__wave_id, fname_recogres, fname_wordhist, 1 )

        # send foward_message
        self.set_forward_msg( np.array(self.__obj_hist,dtype=np.float) )


    def sample_index(self, P):
        K = len(P)
        for z in range(1,K):
            P[z] = P[z] + P[z-1]

        # サンプリング
        rnd = P[K-1] * random.random()
        for z in range(K):
            if P[z] >= rnd:
                return z

        return -1


    def select_senteces(self, fname_recog, fname_selected_sen):
        prob = self.get_backward_msg()
        sentences = utils.load_lines( fname_recog, str )

        selected_sen = []

        for i in range(0, len(sentences), self.__nbest):
            object_id = self.__wave_id[i][0]
            logliks = np.sum( self.__sen_hist[i:i+self.__nbest] * np.log(prob[object_id]), 1 )
            liks = np.exp( logliks - np.max(logliks) )
            
            # ヒストグラムが空の時尤度が高くなってしまう場合があるので，その対処
            liks = liks * ( np.sum(self.__sen_hist[i:i+self.__nbest], 1)>0 )
            idx = self.sample_index( liks )
            selected_sen.append( sentences[i+idx] )

        utils.save_lines( selected_sen, fname_selected_sen )


    def learn(self, nit):
        fname_selected_sen = os.path.join( self.get_name(), "%03d"%nit, "selected_sentences.txt" )
        fname_recog = os.path.join( self.get_name(), "%03d"%(nit-1), "recogres.txt" )
#        fname_recog0 = os.path.join( self.get_name(), "000", "recogres.txt" )
        fname_lm = os.path.join( self.get_name(), "%03d"%nit, "lm" )
#        fname_sentences_lm = os.path.join( self.get_name(), "%03d"%nit, "sentences_for_lm.txt" )

        # backwardメッセージを使用して，音声認識結果をリサンプリング
        self.select_senteces( fname_recog, fname_selected_sen )
        
        # 選択された認識結果から言語モデルを更新
#        lang_model.combine_files( [fname_recog0] + [fname_selected_sen], fname_sentences_lm )
#        lang_model.makelm( fname_sentences_lm, fname_lm )
        lang_model.makelm( fname_selected_sen, fname_lm )

        fname_recogres = os.path.join( self.get_name(), "%03d"%nit, "recogres.txt" )
        fname_best = os.path.join( self.get_name(), "%03d"%nit, "best_sentences.txt" )
        fname_wordhist = os.path.join( self.get_name(), "%03d"%nit, "word" )

        htkdic = fname_lm + ".htkdic"
        bingram = fname_lm + ".bingram"

        # 更新した言語モデルで音声認識
        self.__wave_id = []
        sentences = []
        best_sentences = []
        for d in range(len(self.__wave_files)):
            for n in range(len(self.__wave_files[d])):
                recogres = self.__julius.recog( self.__wave_files[d][n], self.__nbest, bingram, htkdic )
                best_sentences.append( recogres[0].replace("ー", "") )
                for i in range(self.__nbest):
                    self.__wave_id.append((d,n,i))
                    if i < len(recogres):
                        sentences.append( recogres[i].replace("ー", "") )
                    else:
                        sentences.append("")

        utils.save_lines( sentences, fname_recogres )
        utils.save_lines( best_sentences, fname_best )
        self.__obj_hist, self.__sen_hist = histogram.make_histogram( self.__wave_id, fname_recogres, fname_wordhist, self.__threshold )

        # ヒストグラムを送信
        self.set_forward_msg( np.array(self.__obj_hist, dtype=np.float) )


    def update(self):
        if self.__nit==0:
            mkdir( os.path.join( self.get_name(), "000" ) )
            self.initialize()
            self.__nit += 1
        else:
            mkdir( os.path.join( self.get_name(), "%03d"%self.__nit ) )
            self.learn( self.__nit )
            self.__nit += 1
#            if self.__threshold < 100:
#                self.__threshold *= 2
