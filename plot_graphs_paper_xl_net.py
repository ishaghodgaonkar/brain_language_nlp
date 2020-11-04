from subprocess import run

for i in range(0, 40, 5):
    if i == 0:
        i+=1
    string_to_execute = "nohup python3 extract_nlp_features.py --nlp_model xl_net --sequence_length " + str(i) + " --output_dir nlp_features"

    run(string_to_execute.split())
    run(["disown"])

# predict brain from nlp
for layer in range(0, 13):
    string_to_execute = "nohup python3 predict_brain_from_nlp.py --subject F --nlp_feat_type xl_net --nlp_feat_dir nlp_features/ --layer --sequence_length " + str(layer) + " --output_dir encoding_model/ &"

    run(string_to_execute.split())
    run(["disown"])

# evaluate predictions
for layer in range(0, 13):
    string_to_execute = "nohup python3 evaluate_brain_predictions.py --input_path encoding_model/predict_F_with_xl_net_layer_" + str(layer) + "_len_40.npy --output_path predictions/ --subject F &"
    run(string_to_execute.split())
    run(["disown"])



