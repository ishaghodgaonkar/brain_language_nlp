from subprocess import run
# extract features
for i in range(0, 40, 5):
    if i == 0:
        i+=1
    string_to_execute = "python3 extract_nlp_features.py --nlp_model bert --sequence_length " + str(i) + " --output_dir nlp_features/ "

    run(string_to_execute.split())

# predict brain from nlp
for layer in range(0, 13):
    for seq_len in range(0, 40, 5):
        if seq_len == 0:
            seq_len+=1
        string_to_execute = "python3 predict_brain_from_nlp.py --subject F --nlp_feat_type bert --nlp_feat_dir nlp_features/ --layer " + str(layer) + "  --sequence_length "  + str(seq_len) + " --output_dir /local/a/cam2/isha/encoding_model/ "
        run(string_to_execute.split())

# evaluate predictions
for layer in range(0, 13):
    for seq_len in range(0, 40, 5):
        if seq_len == 0:
            seq_len+=1
        string_to_execute = "python3 evaluate_brain_predictions.py --input_path encoding_model/predict_F_with_bert_layer_" + str(layer) + "_len_" + str(seq_len) + ".npy --output_path /local/a/cam2/isha/predictions/ --subject F "
        run(string_to_execute.split())



