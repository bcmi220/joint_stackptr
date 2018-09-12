import sys
import os
#import ufal.udpipe

language_name_to_short = [
	{'name':'Afrikaans-AfriBooms','key':'afrikaans-afribooms','dev':'af_afribooms','train':'af_afribooms','udpipe':'True'},
	{'name':'Ancient_Greek-Perseus','key':'ancient_greek-perseus','dev':'grc_perseus','train':'grc_perseus','udpipe':'True'},
	{'name':'Ancient_Greek-PROIEL','key':'ancient_greek-proiel','dev':'grc_proiel','train':'grc_proiel','udpipe':'True'},
	{'name':'Arabic-PADT','key':'arabic-padt','dev':'ar_padt','train':'ar_padt','udpipe':'True'},
	{'name':'Armenian-ArmTDP','key':'armenian-armtdp','train':'hy_armtdp','udpipe':'True'},#no dev
	{'name':'Basque-BDT','key':'basque-bdt','dev':'eu_bdt','train':'eu_bdt','udpipe':'True'},
	{'name':'Breton-KEB','key':'breton-keb'},#no train dev
	{'name':'Bulgarian-BTB','key':'bulgarian-btb','dev':'bg_btb','train':'bg_btb','udpipe':'True'},
	{'name':'Buryat-BDT','key':'buryat-bdt','train':'bxr_bdt','udpipe':'True'},#no dev
	{'name':'Catalan-AnCora','key':'catalan-ancora','dev':'ca_ancora','train':'ca_ancora','udpipe':'True'},
	{'name':'Chinese-GSD','key':'chinese-gsd','dev':'zh_gsd','train':'zh_gsd','udpipe':'True'},
	{'name':'Croatian-SET','key':'croatian-set','dev':'hr_set','train':'hr_set','udpipe':'True'},
	{'name':'Czech-CAC','key':'czech-cac','dev':'cs_cac','train':'cs_cac','udpipe':'True'},
	{'name':'Czech-FicTree','key':'czech-fictree','dev':'cs_fictree','train':'cs_fictree','udpipe':'True'},
	{'name':'Czech-PDT','key':'czech-pdt','dev':'cs_pdt','train':'cs_pdt','udpipe':'True'},
	{'name':'Czech-PUD','key':'czech-pud'},#no train dev
	{'name':'Danish-DDT','key':'danish-ddt','dev':'da_ddt','train':'da_ddt','udpipe':'True'},
	{'name':'Dutch-Alpino','key':'dutch-alpino','dev':'nl_alpino','train':'nl_alpino','udpipe':'True'},
	{'name':'Dutch-LassySmall','key':'dutch-lassysmall','dev':'nl_lassysmall','train':'nl_lassysmall','udpipe':'True'},
	{'name':'English-EWT','key':'english-ewt','dev':'en_ewt','train':'en_ewt','udpipe':'True'},
	{'name':'English-GUM','key':'english-gum','dev':'en_gum','train':'en_gum','udpipe':'True'},
	{'name':'English-LinES','key':'english-lines','dev':'en_lines','train':'en_lines','udpipe':'True'},
	{'name':'English-PUD','key':'english-pud'},#no train dev
	{'name':'Estonian-EDT','key':'estonian-edt','dev':'et_edt','train':'et_edt','udpipe':'True'},
	{'name':'Faroese-OFT','key':'faroese-oft'},#no train dev
	{'name':'Finnish-FTB','key':'finnish-ftb','dev':'fi_ftb','train':'fi_ftb','udpipe':'True'},
	{'name':'Finnish-PUD','key':'finnish-pud'},# no train dev
	{'name':'Finnish-TDT','key':'finnish-tdt','dev':'fi_tdt','train':'fi_tdt','udpipe':'True'},
	{'name':'French-GSD','key':'french-gsd','dev':'fr_gsd','train':'fr_gsd','udpipe':'True'},
	{'name':'French-Sequoia','key':'french-sequoia','dev':'fr_sequoia','train':'fr_sequoia','udpipe':'True'},
	{'name':'French-Spoken','key':'french-spoken','dev':'fr_spoken','train':'fr_spoken','udpipe':'True'},
	{'name':'Galician-CTG','key':'galician-ctg','dev':'gl_ctg','train':'gl_ctg','udpipe':'True'},
	{'name':'Galician-TreeGal','key':'galician-treegal','train':'gl_treegal','udpipe':'True'},# no dev
	{'name':'German-GSD','key':'german-gsd','dev':'de_gsd','train':'de_gsd','udpipe':'True'},
	{'name':'Gothic-PROIEL','key':'gothic-proiel','dev':'got_proiel','train':'got_proiel','udpipe':'True'},
	{'name':'Greek-GDT','key':'greek-gdt','dev':'el_gdt','train':'el_gdt','udpipe':'True'},
	{'name':'Hebrew-HTB','key':'hebrew-htb','dev':'he_htb','train':'he_htb','udpipe':'True'},
	{'name':'Hindi-HDTB','key':'hindi-hdtb','dev':'hi_hdtb','train':'hi_hdtb','udpipe':'True'},
	{'name':'Hungarian-Szeged','key':'hungarian-szeged','dev':'hu_szeged','train':'hu_szeged','udpipe':'True'},
	{'name':'Indonesian-GSD','key':'indonesian-gsd','dev':'id_gsd','train':'id_gsd','udpipe':'True'},
	{'name':'Irish-IDT','key':'irish-idt','train':'ga_idt','udpipe':'True'},# no dev
	{'name':'Italian-ISDT','key':'italian-isdt','dev':'it_isdt','train':'it_isdt','udpipe':'True'},
	{'name':'Italian-PoSTWITA','key':'italian-postwita','dev':'it_postwita','train':'it_postwita','udpipe':'True'},
	{'name':'Japanese-GSD','key':'japanese-gsd','dev':'ja_gsd','train':'ja_gsd','udpipe':'True'},
	{'name':'Japanese-Modern','key':'japanese-modern'},# no train dev
	{'name':'Kazakh-KTB','key':'kazakh-ktb','train':'kk_ktb','udpipe':'True'},#no dev
	{'name':'Korean-GSD','key':'korean-gsd','dev':'ko_gsd','train':'ko_gsd','udpipe':'True'},
	{'name':'Korean-Kaist','key':'korean-kaist','dev':'ko_kaist','train':'ko_kaist','udpipe':'True'},
	{'name':'Kurmanji-MG','key':'kurmanji-mg','train':'kmr_mg','udpipe':'True'},# no dev
	{'name':'Latin-ITTB','key':'latin-ittb','dev':'la_ittb','train':'la_ittb','udpipe':'True'},
	{'name':'Latin-Perseus','key':'latin-perseus','train':'la_perseus','udpipe':'True'},# no dev
	{'name':'Latin-PROIEL','key':'latin-proiel','dev':'la_proiel','train':'la_proiel','udpipe':'True'},
	{'name':'Latvian-LVTB','key':'latvian-lvtb','dev':'lv_lvtb','train':'lv_lvtb','udpipe':'True'},
	{'name':'Naija-NSC','key':'naija-nsc'},# no train dev
	{'name':'North_Sami-Giella','key':'north_sami-giella','train':'sme_giella','udpipe':'True'},# no dev
	{'name':'Norwegian-Bokmaal','key':'norwegian-bokmaal','dev':'no_bokmaal','train':'no_bokmaal','udpipe':'True'},
	{'name':'Norwegian-Nynorsk','key':'norwegian-nynorsk','dev':'no_nynorsk','train':'no_nynorsk','udpipe':'True'},
	{'name':'Norwegian-NynorskLIA','key':'norwegian-nynorsklia','train':'no_nynorsklia','udpipe':'True'},# no dev
	{'name':'Old_Church_Slavonic-PROIEL','key':'old_church_slavonic-proiel','dev':'cu_proiel','train':'cu_proiel','udpipe':'True'},
	{'name':'Old_French-SRCMF','key':'old_french-srcmf','dev':'fro_srcmf','train':'fro_srcmf','udpipe':'True'},
	{'name':'Persian-Seraji','key':'persian-seraji','dev':'fa_seraji','train':'fa_seraji','udpipe':'True'},
	{'name':'Polish-LFG','key':'polish-lfg','dev':'pl_lfg','train':'pl_lfg','udpipe':'True'},
	{'name':'Polish-SZ','key':'polish-sz','dev':'pl_sz','train':'pl_sz','udpipe':'True'},
	{'name':'Portuguese-Bosque','key':'portuguese-bosque','dev':'pt_bosque','train':'pt_bosque','udpipe':'True'},
	{'name':'Romanian-RRT','key':'romanian-rrt','dev':'ro_rrt','train':'ro_rrt','udpipe':'True'},
	{'name':'Russian-SynTagRus','key':'russian-syntagrus','dev':'ru_syntagrus','train':'ru_syntagrus','udpipe':'True'},
	{'name':'Russian-Taiga','key':'russian-taiga','train':'ru_taiga','udpipe':'True'},# no dev
	{'name':'Serbian-SET','key':'serbian-set','dev':'sr_set','train':'sr_set','udpipe':'True'},
	{'name':'Slovak-SNK','key':'slovak-snk','dev':'sk_snk','train':'sk_snk','udpipe':'True'},
	{'name':'Slovenian-SSJ','key':'slovenian-ssj','dev':'sl_ssj','train':'sl_ssj','udpipe':'True'},
	{'name':'Slovenian-SST','key':'slovenian-sst','train':'sl_sst','udpipe':'True'},#no dev
	{'name':'Spanish-AnCora','key':'spanish-ancora','dev':'es_ancora','train':'es_ancora','udpipe':'True'},
	{'name':'Swedish-LinES','key':'swedish-lines','dev':'sv_lines','train':'sv_lines','udpipe':'True'},
	{'name':'Swedish-PUD','key':'swedish-pud'},# no train dev
	{'name':'Swedish-Talbanken','key':'swedish-talbanken','dev':'sv_talbanken','train':'sv_talbanken','udpipe':'True'},
	{'name':'Thai-PUD','key':'thai-pud'},# no train dev
	{'name':'Turkish-IMST','key':'turkish-imst','dev':'tr_imst','train':'tr_imst','udpipe':'True'},
	{'name':'Ukrainian-IU','key':'ukrainian-iu','dev':'uk_iu','train':'uk_iu','udpipe':'True'},
	{'name':'Upper_Sorbian-UFAL','key':'upper_sorbian-ufal','train':'hsb_ufal','udpipe':'True'},# no dev
	{'name':'Urdu-UDTB','key':'urdu-udtb','dev':'ur_udtb','train':'ur_udtb','udpipe':'True'},
	{'name':'Uyghur-UDT','key':'uyghur-udt','dev':'ug_udt','train':'ug_udt','udpipe':'True'},
	{'name':'Vietnamese-VTB','key':'vietnamese-vtb','dev':'vi_vtb','train':'vi_vtb','udpipe':'True'},
]





udpipe_model = []
for file in os.listdir('../../../UDPipe/ud-2.2-conll18-baseline-models/models'):
    if file.endswith('.udpipe'):
        udpipe_model.append(file[:len(file)-29])

print(udpipe_model)
print(len(udpipe_model))

count = 0
dev_count = 0
train_count = 0
all_files = []
udpipe_count = 0
print('[')
for file in os.listdir('../data/ud-treebanks-v2.2'):
    name_dict = {}
    name_dict['name'] = file[3:]
    name_dict['key'] = file[3:].lower()
    if name_dict['key'] in udpipe_model:
        name_dict['udpipe'] = 'True'
        udpipe_count += 1
    count+=1
    for subfile in os.listdir('../data/ud-treebanks-v2.2/'+file+'/'):
        if subfile.endswith('.conllu') and 'dev' in subfile:
            dev_count+=1
            name_dict['dev'] = subfile.split('-')[0]
        if subfile.endswith('.conllu') and 'train' in subfile:
            train_count+=1
            name_dict['train'] = subfile.split('-')[0]
            name_dict['lcode'],name_dict['tcode'] = name_dict['train'].split('_')
    all_files.append(name_dict)
    print('\t{'+'\'name\':\''+name_dict['name']+'\',\'key\':\''+name_dict['key']+'\'',end='')
    if 'lcode' in  name_dict.keys():
        print(',\'lcode\':\''+name_dict['lcode']+'\'',end='')
    if 'tcode' in  name_dict.keys():
        print(',\'tcode\':\''+name_dict['tcode']+'\'',end='')
    if 'dev' in  name_dict.keys():
        print(',\'dev\':\''+name_dict['dev']+'\'', end='')
    if 'train' in  name_dict.keys():
        print(',\'train\':\''+name_dict['train']+'\'',end='')
    if 'udpipe' in  name_dict.keys():
        print(',\'udpipe\':\''+name_dict['udpipe']+'\'',end='')
    print('},')
print(']')
# print(all_files)
print(count, train_count, dev_count, udpipe_count)









