#!/usr/bin/env bash





CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/af_afribooms.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Afrikaans-AfriBooms" \
--train "data/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms/af_afribooms-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms/af_afribooms-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Afrikaans-AfriBooms/af_afribooms-ud-dev.conllu" \
--model_path "models/ud-parsing/Afrikaans-AfriBooms/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/grc_perseus.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Ancient_Greek-Perseus" \
--train "data/ud-treebanks-v2.2/UD_Ancient_Greek-Perseus/grc_perseus-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.conllu" \
--model_path "models/ud-parsing/Ancient_Greek-Perseus/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/grc_proiel.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Ancient_Greek-PROIEL" \
--train "data/ud-treebanks-v2.2/UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu" \
--model_path "models/ud-parsing/Ancient_Greek-PROIEL/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ar_padt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Arabic-PADT" \
--train "data/ud-treebanks-v2.2/UD_Arabic-PADT/ar_padt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Arabic-PADT/ar_padt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Arabic-PADT/ar_padt-ud-dev.conllu" \
--model_path "models/ud-parsing/Arabic-PADT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/eu_bdt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Basque-BDT" \
--train "data/ud-treebanks-v2.2/UD_Basque-BDT/eu_bdt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Basque-BDT/eu_bdt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Basque-BDT/eu_bdt-ud-dev.conllu" \
--model_path "models/ud-parsing/Basque-BDT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/bg_btb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Bulgarian-BTB" \
--train "data/ud-treebanks-v2.2/UD_Bulgarian-BTB/bg_btb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Bulgarian-BTB/bg_btb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Bulgarian-BTB/bg_btb-ud-dev.conllu" \
--model_path "models/ud-parsing/Bulgarian-BTB/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ca_ancora.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Catalan-AnCora" \
--train "data/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Catalan-AnCora/ca_ancora-ud-dev.conllu" \
--model_path "models/ud-parsing/Catalan-AnCora/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/zh_gsd.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Chinese-GSD" \
--train "data/ud-treebanks-v2.2/UD_Chinese-GSD/zh_gsd-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Chinese-GSD/zh_gsd-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Chinese-GSD/zh_gsd-ud-dev.conllu" \
--model_path "models/ud-parsing/Chinese-GSD/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/hr_set.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Croatian-SET" \
--train "data/ud-treebanks-v2.2/UD_Croatian-SET/hr_set-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Croatian-SET/hr_set-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Croatian-SET/hr_set-ud-dev.conllu" \
--model_path "models/ud-parsing/Croatian-SET/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/cs_cac.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Czech-CAC" \
--train "data/ud-treebanks-v2.2/UD_Czech-CAC/cs_cac-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Czech-CAC/cs_cac-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Czech-CAC/cs_cac-ud-dev.conllu" \
--model_path "models/ud-parsing/Czech-CAC/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/cs_fictree.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Czech-FicTree" \
--train "data/ud-treebanks-v2.2/UD_Czech-FicTree/cs_fictree-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Czech-FicTree/cs_fictree-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Czech-FicTree/cs_fictree-ud-dev.conllu" \
--model_path "models/ud-parsing/Czech-FicTree/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/cs_pdt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Czech-PDT" \
--train "data/ud-treebanks-v2.2/UD_Czech-PDT/cs_pdt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Czech-PDT/cs_pdt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Czech-PDT/cs_pdt-ud-dev.conllu" \
--model_path "models/ud-parsing/Czech-PDT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/da_ddt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Danish-DDT" \
--train "data/ud-treebanks-v2.2/UD_Danish-DDT/da_ddt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Danish-DDT/da_ddt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Danish-DDT/da_ddt-ud-dev.conllu" \
--model_path "models/ud-parsing/Danish-DDT/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/nl_alpino.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Dutch-Alpino" \
--train "data/ud-treebanks-v2.2/UD_Dutch-Alpino/nl_alpino-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Dutch-Alpino/nl_alpino-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Dutch-Alpino/nl_alpino-ud-dev.conllu" \
--model_path "models/ud-parsing/Dutch-Alpino/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/nl_lassysmall.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Dutch-LassySmall" \
--train "data/ud-treebanks-v2.2/UD_Dutch-LassySmall/nl_lassysmall-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Dutch-LassySmall/nl_lassysmall-ud-dev.conllu" \
--model_path "models/ud-parsing/Dutch-LassySmall/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
 --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
 --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
 --schedule 20 --double_schedule_decay 5 \
 --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
 --grandPar --sibling \
 --word_embedding sskip --word_path "data/sskip/en_ewt.skip.forms.50.vectors.gz" --char_embedding random \
 --punctuation '.' '``' "''" ':' ',' \
 --language "English-EWT" \
 --train "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" \
 --dev "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu" \
 --test "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu" \
 --model_path "models/ud-parsing/English-EWT/" --model_name 'network.pt'



CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/en_gum.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "English-GUM" \
--train "data/ud-treebanks-v2.2/UD_English-GUM/en_gum-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_English-GUM/en_gum-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_English-GUM/en_gum-ud-dev.conllu" \
--model_path "models/ud-parsing/English-GUM/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/en_lines.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "English-LinES" \
--train "data/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_English-LinES/en_lines-ud-dev.conllu" \
--model_path "models/ud-parsing/English-LinES/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/et_edt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Estonian-EDT" \
--train "data/ud-treebanks-v2.2/UD_Estonian-EDT/et_edt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Estonian-EDT/et_edt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Estonian-EDT/et_edt-ud-dev.conllu" \
--model_path "models/ud-parsing/Estonian-EDT/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fi_ftb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Finnish-FTB" \
--train "data/ud-treebanks-v2.2/UD_Finnish-FTB/fi_ftb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Finnish-FTB/fi_ftb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Finnish-FTB/fi_ftb-ud-dev.conllu" \
--model_path "models/ud-parsing/Finnish-FTB/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fi_tdt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Finnish-TDT" \
--train "data/ud-treebanks-v2.2/UD_Finnish-TDT/fi_tdt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Finnish-TDT/fi_tdt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Finnish-TDT/fi_tdt-ud-dev.conllu" \
--model_path "models/ud-parsing/Finnish-TDT/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fr_gsd.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "French-GSD" \
--train "data/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_French-GSD/fr_gsd-ud-dev.conllu" \
--model_path "models/ud-parsing/French-GSD/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fr_sequoia.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "French-Sequoia" \
--train "data/ud-treebanks-v2.2/UD_French-Sequoia/fr_sequoia-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_French-Sequoia/fr_sequoia-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_French-Sequoia/fr_sequoia-ud-dev.conllu" \
--model_path "models/ud-parsing/French-Sequoia/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fr_spoken.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "French-Spoken" \
--train "data/ud-treebanks-v2.2/UD_French-Spoken/fr_spoken-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_French-Spoken/fr_spoken-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_French-Spoken/fr_spoken-ud-dev.conllu" \
--model_path "models/ud-parsing/French-Spoken/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/gl_ctg.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Galician-CTG" \
--train "data/ud-treebanks-v2.2/UD_Galician-CTG/gl_ctg-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Galician-CTG/gl_ctg-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Galician-CTG/gl_ctg-ud-dev.conllu" \
--model_path "models/ud-parsing/Galician-CTG/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/de_gsd.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "German-GSD" \
--train "data/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_German-GSD/de_gsd-ud-dev.conllu" \
--model_path "models/ud-parsing/German-GSD/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/got_proiel.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Gothic-PROIEL" \
--train "data/ud-treebanks-v2.2/UD_Gothic-PROIEL/got_proiel-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Gothic-PROIEL/got_proiel-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Gothic-PROIEL/got_proiel-ud-dev.conllu" \
--model_path "models/ud-parsing/Gothic-PROIEL/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/el_gdt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Greek-GDT" \
--train "data/ud-treebanks-v2.2/UD_Greek-GDT/el_gdt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Greek-GDT/el_gdt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Greek-GDT/el_gdt-ud-dev.conllu" \
--model_path "models/ud-parsing/Greek-GDT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/he_htb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Hebrew-HTB" \
--train "data/ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Hebrew-HTB/he_htb-ud-dev.conllu" \
--model_path "models/ud-parsing/Hebrew-HTB/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/hi_hdtb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Hindi-HDTB" \
--train "data/ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Hindi-HDTB/hi_hdtb-ud-dev.conllu" \
--model_path "models/ud-parsing/Hindi-HDTB/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/hu_szeged.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Hungarian-Szeged" \
--train "data/ud-treebanks-v2.2/UD_Hungarian-Szeged/hu_szeged-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Hungarian-Szeged/hu_szeged-ud-dev.conllu" \
--model_path "models/ud-parsing/Hungarian-Szeged/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/id_gsd.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Indonesian-GSD" \
--train "data/ud-treebanks-v2.2/UD_Indonesian-GSD/id_gsd-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Indonesian-GSD/id_gsd-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Indonesian-GSD/id_gsd-ud-dev.conllu" \
--model_path "models/ud-parsing/Indonesian-GSD/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/it_isdt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Italian-ISDT" \
--train "data/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Italian-ISDT/it_isdt-ud-dev.conllu" \
--model_path "models/ud-parsing/Italian-ISDT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/it_postwita.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Italian-PoSTWITA" \
--train "data/ud-treebanks-v2.2/UD_Italian-PoSTWITA/it_postwita-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Italian-PoSTWITA/it_postwita-ud-dev.conllu" \
--model_path "models/ud-parsing/Italian-PoSTWITA/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ja_gsd.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Japanese-GSD" \
--train "data/ud-treebanks-v2.2/UD_Japanese-GSD/ja_gsd-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Japanese-GSD/ja_gsd-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Japanese-GSD/ja_gsd-ud-dev.conllu" \
--model_path "models/ud-parsing/Japanese-GSD/" --model_name 'network.pt'

CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ko_gsd.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Korean-GSD" \
--train "data/ud-treebanks-v2.2/UD_Korean-GSD/ko_gsd-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Korean-GSD/ko_gsd-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Korean-GSD/ko_gsd-ud-dev.conllu" \
--model_path "models/ud-parsing/Korean-GSD/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ko_kaist.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Korean-Kaist" \
--train "data/ud-treebanks-v2.2/UD_Korean-Kaist/ko_kaist-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Korean-Kaist/ko_kaist-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Korean-Kaist/ko_kaist-ud-dev.conllu" \
--model_path "models/ud-parsing/Korean-Kaist/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/la_ittb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Latin-ITTB" \
--train "data/ud-treebanks-v2.2/UD_Latin-ITTB/la_ittb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Latin-ITTB/la_ittb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Latin-ITTB/la_ittb-ud-dev.conllu" \
--model_path "models/ud-parsing/Latin-ITTB/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/la_proiel.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Latin-PROIEL" \
--train "data/ud-treebanks-v2.2/UD_Latin-PROIEL/la_proiel-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Latin-PROIEL/la_proiel-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Latin-PROIEL/la_proiel-ud-dev.conllu" \
--model_path "models/ud-parsing/Latin-PROIEL/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/lv_lvtb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Latvian-LVTB" \
--train "data/ud-treebanks-v2.2/UD_Latvian-LVTB/lv_lvtb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Latvian-LVTB/lv_lvtb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Latvian-LVTB/lv_lvtb-ud-dev.conllu" \
--model_path "models/ud-parsing/Latvian-LVTB/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/no_bokmaal.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Norwegian-Bokmaal" \
--train "data/ud-treebanks-v2.2/UD_Norwegian-Bokmaal/no_bokmaal-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Norwegian-Bokmaal/no_bokmaal-ud-dev.conllu" \
--model_path "models/ud-parsing/Norwegian-Bokmaal/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/no_nynorsk.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Norwegian-Nynorsk" \
--train "data/ud-treebanks-v2.2/UD_Norwegian-Nynorsk/no_nynorsk-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Norwegian-Nynorsk/no_nynorsk-ud-dev.conllu" \
--model_path "models/ud-parsing/Norwegian-Nynorsk/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/cu_proiel.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Old_Church_Slavonic-PROIEL" \
--train "data/ud-treebanks-v2.2/UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Old_Church_Slavonic-PROIEL/cu_proiel-ud-dev.conllu" \
--model_path "models/ud-parsing/Old_Church_Slavonic-PROIEL/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fro_srcmf.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Old_French-SRCMF" \
--train "data/ud-treebanks-v2.2/UD_Old_French-SRCMF/fro_srcmf-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Old_French-SRCMF/fro_srcmf-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Old_French-SRCMF/fro_srcmf-ud-dev.conllu" \
--model_path "models/ud-parsing/Old_French-SRCMF/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/fa_seraji.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Persian-Seraji" \
--train "data/ud-treebanks-v2.2/UD_Persian-Seraji/fa_seraji-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Persian-Seraji/fa_seraji-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Persian-Seraji/fa_seraji-ud-dev.conllu" \
--model_path "models/ud-parsing/Persian-Seraji/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/pl_lfg.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Polish-LFG" \
--train "data/ud-treebanks-v2.2/UD_Polish-LFG/pl_lfg-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Polish-LFG/pl_lfg-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Polish-LFG/pl_lfg-ud-dev.conllu" \
--model_path "models/ud-parsing/Polish-LFG/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/pl_sz.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Polish-SZ" \
--train "data/ud-treebanks-v2.2/UD_Polish-SZ/pl_sz-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Polish-SZ/pl_sz-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Polish-SZ/pl_sz-ud-dev.conllu" \
--model_path "models/ud-parsing/Polish-SZ/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/pt_bosque.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Portuguese-Bosque" \
--train "data/ud-treebanks-v2.2/UD_Portuguese-Bosque/pt_bosque-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu" \
--model_path "models/ud-parsing/Portuguese-Bosque/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ro_rrt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Romanian-RRT" \
--train "data/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Romanian-RRT/ro_rrt-ud-dev.conllu" \
--model_path "models/ud-parsing/Romanian-RRT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ru_syntagrus.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Russian-SynTagRus" \
--train "data/ud-treebanks-v2.2/UD_Russian-SynTagRus/ru_syntagrus-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Russian-SynTagRus/ru_syntagrus-ud-dev.conllu" \
--model_path "models/ud-parsing/Russian-SynTagRus/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/sr_set.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Serbian-SET" \
--train "data/ud-treebanks-v2.2/UD_Serbian-SET/sr_set-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Serbian-SET/sr_set-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Serbian-SET/sr_set-ud-dev.conllu" \
--model_path "models/ud-parsing/Serbian-SET/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/sk_snk.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Slovak-SNK" \
--train "data/ud-treebanks-v2.2/UD_Slovak-SNK/sk_snk-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Slovak-SNK/sk_snk-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Slovak-SNK/sk_snk-ud-dev.conllu" \
--model_path "models/ud-parsing/Slovak-SNK/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/sl_ssj.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Slovenian-SSJ" \
--train "data/ud-treebanks-v2.2/UD_Slovenian-SSJ/sl_ssj-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Slovenian-SSJ/sl_ssj-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Slovenian-SSJ/sl_ssj-ud-dev.conllu" \
--model_path "models/ud-parsing/Slovenian-SSJ/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/es_ancora.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Spanish-AnCora" \
--train "data/ud-treebanks-v2.2/UD_Spanish-AnCora/es_ancora-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Spanish-AnCora/es_ancora-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Spanish-AnCora/es_ancora-ud-dev.conllu" \
--model_path "models/ud-parsing/Spanish-AnCora/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/sv_lines.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Swedish-LinES" \
--train "data/ud-treebanks-v2.2/UD_Swedish-LinES/sv_lines-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Swedish-LinES/sv_lines-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Swedish-LinES/sv_lines-ud-dev.conllu" \
--model_path "models/ud-parsing/Swedish-LinES/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/sv_talbanken.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Swedish-Talbanken" \
--train "data/ud-treebanks-v2.2/UD_Swedish-Talbanken/sv_talbanken-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Swedish-Talbanken/sv_talbanken-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Swedish-Talbanken/sv_talbanken-ud-dev.conllu" \
--model_path "models/ud-parsing/Swedish-Talbanken/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/tr_imst.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Turkish-IMST" \
--train "data/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Turkish-IMST/tr_imst-ud-dev.conllu" \
--model_path "models/ud-parsing/Turkish-IMST/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/uk_iu.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Ukrainian-IU" \
--train "data/ud-treebanks-v2.2/UD_Ukrainian-IU/uk_iu-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Ukrainian-IU/uk_iu-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Ukrainian-IU/uk_iu-ud-dev.conllu" \
--model_path "models/ud-parsing/Ukrainian-IU/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ur_udtb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Urdu-UDTB" \
--train "data/ud-treebanks-v2.2/UD_Urdu-UDTB/ur_udtb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Urdu-UDTB/ur_udtb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Urdu-UDTB/ur_udtb-ud-dev.conllu" \
--model_path "models/ud-parsing/Urdu-UDTB/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/ug_udt.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Uyghur-UDT" \
--train "data/ud-treebanks-v2.2/UD_Uyghur-UDT/ug_udt-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Uyghur-UDT/ug_udt-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Uyghur-UDT/ug_udt-ud-dev.conllu" \
--model_path "models/ud-parsing/Uyghur-UDT/" --model_name 'network.pt'


CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 100 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
--pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
--schedule 20 --double_schedule_decay 5 \
--p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
--grandPar --sibling \
--word_embedding sskip --word_path "data/sskip/vi_vtb.skip.forms.50.vectors.gz" --char_embedding random \
--punctuation '.' '``' "''" ':' ',' \
--language "Vietnamese-VTB" \
--train "data/ud-treebanks-v2.2/UD_Vietnamese-VTB/vi_vtb-ud-train.conllu" \
--dev "data/ud-treebanks-v2.2/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu" \
--test "data/ud-treebanks-v2.2/UD_Vietnamese-VTB/vi_vtb-ud-dev.conllu" \
--model_path "models/ud-parsing/Vietnamese-VTB/" --model_name 'network.pt'

# CUDA_VISIBLE_DEVICES=0 python examples/StackPointerParser_CoNLLU.py --mode FastLSTM --num_epochs 200 --batch_size 32 --decoder_input_size 256 --hidden_size 512 --encoder_layers 3 --decoder_layers 1 \
# --pos_dim 100 --char_dim 100 --num_filters 100 --arc_space 512 --type_space 128 \
# --opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --coverage 0.0 --gamma 0.0 --clip 5.0 \
# --schedule 20 --double_schedule_decay 5 \
# --p_in 0.33 --p_out 0.33 --p_rnn 0.33 0.33 --unk_replace 0.5 --label_smooth 1.0 --pos --char --beam 1 --prior_order inside_out \
# --grandPar --sibling \
# --word_embedding sskip --word_path "data/sskip/en_ewt.skip.forms.50.vectors.gz" --char_embedding random \
# --punctuation '.' '``' "''" ':' ',' \
# --language "English-EWT" \
# --train "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-train.conllu" \
# --dev "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu" \
# --test "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu" \
# --model_path "models/ud-parsing/English-EWT/" --model_name 'network.pt' \
# --predict \
# --eval "data/ud-treebanks-v2.2/UD_English-EWT/en_ewt-ud-dev.conllu"



