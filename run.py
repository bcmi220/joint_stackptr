#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys, getopt, os
from ufal.udpipe import Model, Pipeline, ProcessingError
import json

if sys.version_info[0] > 2:
    # py3k
    pass
else:
    # py2
    import codecs
    import warnings
    def open(file, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None, closefd=True, opener=None):
        if newline is not None:
            warnings.warn('newline is not supported in py2')
        if not closefd:
            warnings.warn('closefd is not supported in py2')
        if opener is not None:
            warnings.warn('opener is not supported in py2')
        return codecs.open(filename=file, mode=mode, encoding=encoding,
                    errors=errors, buffering=buffering)

DEBUG = True

def get_input_params(argv):
    inputs = ''
    outputs = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["help", "input=","output="])
    except getopt.GetoptError:
        print 'run.py -i <input> -o <output>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'run.py -i <input> -o <output>'
            sys.exit()
        elif opt in ("-i", "--input"):
            inputs = arg
        elif opt in ("-o", "--output"):
            outputs = arg
    return inputs,outputs


def get_model(lcode, tcode):
    model_data = [
        {'name':'Afrikaans-AfriBooms','key':'afrikaans-afribooms','lcode':'af','tcode':'afribooms','dev':'af_afribooms','train':'af_afribooms','udpipe':'True'},
        {'name':'Ancient_Greek-Perseus','key':'ancient_greek-perseus','lcode':'grc','tcode':'perseus','dev':'grc_perseus','train':'grc_perseus','udpipe':'True'},
        {'name':'Ancient_Greek-PROIEL','key':'ancient_greek-proiel','lcode':'grc','tcode':'proiel','dev':'grc_proiel','train':'grc_proiel','udpipe':'True'},
        {'name':'Arabic-PADT','key':'arabic-padt','lcode':'ar','tcode':'padt','dev':'ar_padt','train':'ar_padt','udpipe':'True'},
        {'name':'Armenian-ArmTDP','key':'armenian-armtdp','lcode':'hy','tcode':'armtdp','train':'hy_armtdp','udpipe':'True'},
        {'name':'Basque-BDT','key':'basque-bdt','lcode':'eu','tcode':'bdt','dev':'eu_bdt','train':'eu_bdt','udpipe':'True'},
        {'name':'Breton-KEB','key':'breton-keb'},#
        {'name':'Bulgarian-BTB','key':'bulgarian-btb','lcode':'bg','tcode':'btb','dev':'bg_btb','train':'bg_btb','udpipe':'True'},
        {'name':'Buryat-BDT','key':'buryat-bdt','lcode':'bxr','tcode':'bdt','train':'bxr_bdt','udpipe':'True'},
        {'name':'Catalan-AnCora','key':'catalan-ancora','lcode':'ca','tcode':'ancora','dev':'ca_ancora','train':'ca_ancora','udpipe':'True'},
        {'name':'Chinese-GSD','key':'chinese-gsd','lcode':'zh','tcode':'gsd','dev':'zh_gsd','train':'zh_gsd','udpipe':'True'},
        {'name':'Croatian-SET','key':'croatian-set','lcode':'hr','tcode':'set','dev':'hr_set','train':'hr_set','udpipe':'True'},
        {'name':'Czech-CAC','key':'czech-cac','lcode':'cs','tcode':'cac','dev':'cs_cac','train':'cs_cac','udpipe':'True'},
        {'name':'Czech-FicTree','key':'czech-fictree','lcode':'cs','tcode':'fictree','dev':'cs_fictree','train':'cs_fictree','udpipe':'True'},
        {'name':'Czech-PDT','key':'czech-pdt','lcode':'cs','tcode':'pdt','dev':'cs_pdt','train':'cs_pdt','udpipe':'True'},
        {'name':'Czech-PUD','key':'czech-pud'},#
        {'name':'Danish-DDT','key':'danish-ddt','lcode':'da','tcode':'ddt','dev':'da_ddt','train':'da_ddt','udpipe':'True'},
        {'name':'Dutch-Alpino','key':'dutch-alpino','lcode':'nl','tcode':'alpino','dev':'nl_alpino','train':'nl_alpino','udpipe':'True'},
        {'name':'Dutch-LassySmall','key':'dutch-lassysmall','lcode':'nl','tcode':'lassysmall','dev':'nl_lassysmall','train':'nl_lassysmall','udpipe':'True'},
        {'name':'English-EWT','key':'english-ewt','lcode':'en','tcode':'ewt','dev':'en_ewt','train':'en_ewt','udpipe':'True'},
        {'name':'English-GUM','key':'english-gum','lcode':'en','tcode':'gum','dev':'en_gum','train':'en_gum','udpipe':'True'},
        {'name':'English-LinES','key':'english-lines','lcode':'en','tcode':'lines','dev':'en_lines','train':'en_lines','udpipe':'True'},
        {'name':'English-PUD','key':'english-pud'},#
        {'name':'Estonian-EDT','key':'estonian-edt','lcode':'et','tcode':'edt','dev':'et_edt','train':'et_edt','udpipe':'True'},
        {'name':'Faroese-OFT','key':'faroese-oft'},#
        {'name':'Finnish-FTB','key':'finnish-ftb','lcode':'fi','tcode':'ftb','dev':'fi_ftb','train':'fi_ftb','udpipe':'True'},
        {'name':'Finnish-PUD','key':'finnish-pud'},#
        {'name':'Finnish-TDT','key':'finnish-tdt','lcode':'fi','tcode':'tdt','dev':'fi_tdt','train':'fi_tdt','udpipe':'True'},
        {'name':'French-GSD','key':'french-gsd','lcode':'fr','tcode':'gsd','dev':'fr_gsd','train':'fr_gsd','udpipe':'True'},
        {'name':'French-Sequoia','key':'french-sequoia','lcode':'fr','tcode':'sequoia','dev':'fr_sequoia','train':'fr_sequoia','udpipe':'True'},
        {'name':'French-Spoken','key':'french-spoken','lcode':'fr','tcode':'spoken','dev':'fr_spoken','train':'fr_spoken','udpipe':'True'},
        {'name':'Galician-CTG','key':'galician-ctg','lcode':'gl','tcode':'ctg','dev':'gl_ctg','train':'gl_ctg','udpipe':'True'},
        {'name':'Galician-TreeGal','key':'galician-treegal','lcode':'gl','tcode':'treegal','train':'gl_treegal','udpipe':'True'},
        {'name':'German-GSD','key':'german-gsd','lcode':'de','tcode':'gsd','dev':'de_gsd','train':'de_gsd','udpipe':'True'},
        {'name':'Gothic-PROIEL','key':'gothic-proiel','lcode':'got','tcode':'proiel','dev':'got_proiel','train':'got_proiel','udpipe':'True'},
        {'name':'Greek-GDT','key':'greek-gdt','lcode':'el','tcode':'gdt','dev':'el_gdt','train':'el_gdt','udpipe':'True'},
        {'name':'Hebrew-HTB','key':'hebrew-htb','lcode':'he','tcode':'htb','dev':'he_htb','train':'he_htb','udpipe':'True'},
        {'name':'Hindi-HDTB','key':'hindi-hdtb','lcode':'hi','tcode':'hdtb','dev':'hi_hdtb','train':'hi_hdtb','udpipe':'True'},
        {'name':'Hungarian-Szeged','key':'hungarian-szeged','lcode':'hu','tcode':'szeged','dev':'hu_szeged','train':'hu_szeged','udpipe':'True'},
        {'name':'Indonesian-GSD','key':'indonesian-gsd','lcode':'id','tcode':'gsd','dev':'id_gsd','train':'id_gsd','udpipe':'True'},
        {'name':'Irish-IDT','key':'irish-idt','lcode':'ga','tcode':'idt','train':'ga_idt','udpipe':'True'},
        {'name':'Italian-ISDT','key':'italian-isdt','lcode':'it','tcode':'isdt','dev':'it_isdt','train':'it_isdt','udpipe':'True'},
        {'name':'Italian-PoSTWITA','key':'italian-postwita','lcode':'it','tcode':'postwita','dev':'it_postwita','train':'it_postwita','udpipe':'True'},
        {'name':'Japanese-GSD','key':'japanese-gsd','lcode':'ja','tcode':'gsd','dev':'ja_gsd','train':'ja_gsd','udpipe':'True'},
        {'name':'Japanese-Modern','key':'japanese-modern'},#
        {'name':'Kazakh-KTB','key':'kazakh-ktb','lcode':'kk','tcode':'ktb','train':'kk_ktb','udpipe':'True'},
        {'name':'Korean-GSD','key':'korean-gsd','lcode':'ko','tcode':'gsd','dev':'ko_gsd','train':'ko_gsd','udpipe':'True'},
        {'name':'Korean-Kaist','key':'korean-kaist','lcode':'ko','tcode':'kaist','dev':'ko_kaist','train':'ko_kaist','udpipe':'True'},
        {'name':'Kurmanji-MG','key':'kurmanji-mg','lcode':'kmr','tcode':'mg','train':'kmr_mg','udpipe':'True'},
        {'name':'Latin-ITTB','key':'latin-ittb','lcode':'la','tcode':'ittb','dev':'la_ittb','train':'la_ittb','udpipe':'True'},
        {'name':'Latin-Perseus','key':'latin-perseus','lcode':'la','tcode':'perseus','train':'la_perseus','udpipe':'True'},
        {'name':'Latin-PROIEL','key':'latin-proiel','lcode':'la','tcode':'proiel','dev':'la_proiel','train':'la_proiel','udpipe':'True'},
        {'name':'Latvian-LVTB','key':'latvian-lvtb','lcode':'lv','tcode':'lvtb','dev':'lv_lvtb','train':'lv_lvtb','udpipe':'True'},
        {'name':'Naija-NSC','key':'naija-nsc'},#
        {'name':'North_Sami-Giella','key':'north_sami-giella','lcode':'sme','tcode':'giella','train':'sme_giella','udpipe':'True'},
        {'name':'Norwegian-Bokmaal','key':'norwegian-bokmaal','lcode':'no','tcode':'bokmaal','dev':'no_bokmaal','train':'no_bokmaal','udpipe':'True'},
        {'name':'Norwegian-Nynorsk','key':'norwegian-nynorsk','lcode':'no','tcode':'nynorsk','dev':'no_nynorsk','train':'no_nynorsk','udpipe':'True'},
        {'name':'Norwegian-NynorskLIA','key':'norwegian-nynorsklia','lcode':'no','tcode':'nynorsklia','train':'no_nynorsklia','udpipe':'True'},
        {'name':'Old_Church_Slavonic-PROIEL','key':'old_church_slavonic-proiel','lcode':'cu','tcode':'proiel','dev':'cu_proiel','train':'cu_proiel','udpipe':'True'},
        {'name':'Old_French-SRCMF','key':'old_french-srcmf','lcode':'fro','tcode':'srcmf','dev':'fro_srcmf','train':'fro_srcmf','udpipe':'True'},
        {'name':'Persian-Seraji','key':'persian-seraji','lcode':'fa','tcode':'seraji','dev':'fa_seraji','train':'fa_seraji','udpipe':'True'},
        {'name':'Polish-LFG','key':'polish-lfg','lcode':'pl','tcode':'lfg','dev':'pl_lfg','train':'pl_lfg','udpipe':'True'},
        {'name':'Polish-SZ','key':'polish-sz','lcode':'pl','tcode':'sz','dev':'pl_sz','train':'pl_sz','udpipe':'True'},
        {'name':'Portuguese-Bosque','key':'portuguese-bosque','lcode':'pt','tcode':'bosque','dev':'pt_bosque','train':'pt_bosque','udpipe':'True'},
        {'name':'Romanian-RRT','key':'romanian-rrt','lcode':'ro','tcode':'rrt','dev':'ro_rrt','train':'ro_rrt','udpipe':'True'},
        {'name':'Russian-SynTagRus','key':'russian-syntagrus','lcode':'ru','tcode':'syntagrus','dev':'ru_syntagrus','train':'ru_syntagrus','udpipe':'True'},
        {'name':'Russian-Taiga','key':'russian-taiga','lcode':'ru','tcode':'taiga','train':'ru_taiga','udpipe':'True'},
        {'name':'Serbian-SET','key':'serbian-set','lcode':'sr','tcode':'set','dev':'sr_set','train':'sr_set','udpipe':'True'},
        {'name':'Slovak-SNK','key':'slovak-snk','lcode':'sk','tcode':'snk','dev':'sk_snk','train':'sk_snk','udpipe':'True'},
        {'name':'Slovenian-SSJ','key':'slovenian-ssj','lcode':'sl','tcode':'ssj','dev':'sl_ssj','train':'sl_ssj','udpipe':'True'},
        {'name':'Slovenian-SST','key':'slovenian-sst','lcode':'sl','tcode':'sst','train':'sl_sst','udpipe':'True'},
        {'name':'Spanish-AnCora','key':'spanish-ancora','lcode':'es','tcode':'ancora','dev':'es_ancora','train':'es_ancora','udpipe':'True'},
        {'name':'Swedish-LinES','key':'swedish-lines','lcode':'sv','tcode':'lines','dev':'sv_lines','train':'sv_lines','udpipe':'True'},
        {'name':'Swedish-PUD','key':'swedish-pud'},#
        {'name':'Swedish-Talbanken','key':'swedish-talbanken','lcode':'sv','tcode':'talbanken','dev':'sv_talbanken','train':'sv_talbanken','udpipe':'True'},
        {'name':'Thai-PUD','key':'thai-pud'},#
        {'name':'Turkish-IMST','key':'turkish-imst','lcode':'tr','tcode':'imst','dev':'tr_imst','train':'tr_imst','udpipe':'True'},
        {'name':'Ukrainian-IU','key':'ukrainian-iu','lcode':'uk','tcode':'iu','dev':'uk_iu','train':'uk_iu','udpipe':'True'},
        {'name':'Upper_Sorbian-UFAL','key':'upper_sorbian-ufal','lcode':'hsb','tcode':'ufal','train':'hsb_ufal','udpipe':'True'},
        {'name':'Urdu-UDTB','key':'urdu-udtb','lcode':'ur','tcode':'udtb','dev':'ur_udtb','train':'ur_udtb','udpipe':'True'},
        {'name':'Uyghur-UDT','key':'uyghur-udt','lcode':'ug','tcode':'udt','dev':'ug_udt','train':'ug_udt','udpipe':'True'},
        {'name':'Vietnamese-VTB','key':'vietnamese-vtb','lcode':'vi','tcode':'vtb','dev':'vi_vtb','train':'vi_vtb','udpipe':'True'},
    ]

    for item in model_data:
        if item.get('lcode')==lcode and item.get('tcode')==tcode and (item.get('dev') is not None or item.get('udpipe')=='True'):
            return item
    
    for item in model_data:
        if item.get('lcode')==lcode and (item.get('dev') is not None or item.get('udpipe')=='True'):
            return item
        
    if lcode == 'breton':
        return {'name':'French-GSD','key':'french-gsd','lcode':'fr','tcode':'gsd','dev':'fr_gsd','train':'fr_gsd','udpipe':'True'}

    if lcode == 'faroese':
        return {'name':'Danish-DDT','key':'danish-ddt','lcode':'da','tcode':'ddt','dev':'da_ddt','train':'da_ddt','udpipe':'True'}

    return {'name':'English-EWT','key':'english-ewt','lcode':'en','tcode':'ewt','dev':'en_ewt','train':'en_ewt','udpipe':'True'}


if __name__ == '__main__':
    input_dir, output_dir = get_input_params(sys.argv[1:])

    # if DEBUG:
    #     print('input:',input_dir,'output:',output_dir)

    # 
    with open(os.path.join(input_dir,'metadata.json'), 'r') as f:
        metadata = json.load(f)

    for item in metadata:
        lcode = item['lcode']
        tcode = item['tcode']
        model_params = get_model(lcode, tcode)

        input_file = os.path.join(input_dir, item['psegmorfile'])
        output_file = os.path.join(output_dir, item['outfile'])

        if model_params.get('dev') is not None:
            model = Model.load('models/udpipe/'+model_params.get('key')+'-ud-2.2-conll18-180430.udpipe')
            pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
            error = ProcessingError()

            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = f.readlines()

            udpipe_pred_file = os.path.join('result', model_params['dev']+'.pred')

            with open(udpipe_pred_file, 'w', encoding='utf-8') as f:
                for line in input_data:
                    if len(line.strip())>0 and line.strip()[:9]=='# text = ':
                        text = line[9:]
                        processed = pipeline.process(text, error)
                        if error.occurred():
                            sys.stderr.write("An error occurred when running run_udpipe: ")
                            sys.stderr.write(error.message)
                            sys.stderr.write("\n")
                            sys.exit(1)
                        f.write(processed)
                        f.write('\n')

            input_file = udpipe_pred_file
            
            command = '''python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
                    --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
                    --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
                    --schedule 20 --double_schedule_decay 5 \
                    --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
                    --grandPar --sibling \
                    --word_embedding sskip --word_path "data/sskip/'''+model_params['dev']+'''.skip.forms.50.vectors.gz" --char_embedding random \
                    --punctuation '.' '``' "''" ':' ',' \
                    --model_path "models/ud-parsing/'''+model_params['name']+'''/" --model_name "network.pt" \
                    --predict \
                    --eval "'''+input_file+'''" --output_path "''' + output_file + '"'
            # if DEBUG:
            #     print(command)

            os.system(command)

        elif model_params.get('udpipe') == 'True':
            print('models/udpipe/'+model_params.get('key')+'-ud-2.2-conll18-180430.udpipe')
            model = Model.load('models/udpipe/'+model_params.get('key')+'-ud-2.2-conll18-180430.udpipe')
            pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
            error = ProcessingError()

            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = f.readlines()

            with open(output_file, 'w', encoding='utf-8') as f:
                for line in input_data:
                    if len(line.strip())>0 and line.strip()[:9]=='# text = ':
                        text = line[9:]
                        processed = pipeline.process(text, error)
                        if error.occurred():
                            sys.stderr.write("An error occurred when running run_udpipe: ")
                            sys.stderr.write(error.message)
                            sys.stderr.write("\n")
                            sys.exit(1)
                        f.write(processed)
                        f.write('\n')
        else:
            sys.stderr.write('Unknow language:'+lcode+'-'+tcode)


