import torch.nn

from models import LSTM_text_emotions, CustomBert
from dataset_loading import *
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from audio_learning import scuff_accuracy, bert_accuracy
from utils.funcs import embedding_lookup
from transformers import BertModel, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def main():
    model = LSTM_text_emotions()
    word_id_mapping, word_embedding = load_w2v(300, None, "data_combine_eng/clause_keywords.csv",
                                               "data_combine_eng/model.txt")
    word_embedding = torch.from_numpy(word_embedding)
    train_x, train_y, test_x, test_y, dev_x, dev_y, train_audio, test_audio, dev_audio = load_text_data(word_id_mapping, 41, 30, 7)
    # train_x, train_y, test_x, test_y, dev_x, dev_y = load_text_data_bert(30, 7)
    loader = DataLoader(list(zip(train_x, train_y, train_audio)), shuffle=True, batch_size=32)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_f = torch.nn.MSELoss()
    for epoch in range(1, 100):
        model.train()
        counter = 0
        for tr_x_batch, tr_y_batch, tr_batch_audio in loader:
            optimizer.zero_grad()
            tr_x_batch = embedding_lookup(word_embedding, tr_x_batch)
            tr_pred_y = model(tr_x_batch.to(device), tr_batch_audio.to(device))
            tr_y_batch_f = tr_y_batch.float()
            loss = loss_f(tr_y_batch_f.to(device), tr_pred_y.float())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_x_embed = embedding_lookup(word_embedding, test_x)
            test_pred_y = model(test_x_embed.to(device), test_audio.to(device))
            f1, acc = scuff_accuracy(test_pred_y, test_y)
            print("epoch: " + str(epoch) + "\nacc: " + str(acc) + "\nf1: " + str(f1) + "\n")


def train_bert():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=7,
        output_attentions=False,
        output_hidden_states=False,
    )
    custom_model = CustomBert()
    train_inputs, train_attention, train_labels, test_inputs, test_labels, \
    test_attention, dev_inputs, dev_attention, dev_labels, train_audio, test_audio, dev_audio = load_data_for_bert(30, 7)
    train_dataset = TensorDataset(train_inputs, train_attention, train_labels, train_audio)
    test_dataset = TensorDataset(test_inputs, test_attention, test_labels, test_audio)
    dev_dataset = TensorDataset(dev_inputs, dev_attention, dev_labels, dev_audio)

    loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=16)
    model.to(device)
    custom_model.to(device)
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=2e-5, eps=1e-8)
    loss_f = torch.nn.MSELoss()
    epochs = 15
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    for epoch in range(1, epochs):
        model.train()
        for step, batch in enumerate(loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_audio = batch[3].to(device)
            optimizer.zero_grad()
            output = custom_model(b_input_ids, b_input_mask, b_audio)
            # output = model(b_input_ids,
            #                token_type_ids=None,
            #                attention_mask=b_input_mask,
            #                labels=b_labels)
            # loss = output.loss
            loss = loss_f(b_labels.float(), output.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        total_eval_accuracy = 0
        total_eval_f1 = 0
        model.eval()
        custom_model.eval()
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_audio = batch[3].to(device)
            with torch.no_grad():
                # output = model(b_input_ids,
                #                token_type_ids=None,
                #                attention_mask=b_input_mask,
                #                labels=b_labels)
                output = custom_model(b_input_ids, b_input_mask, b_audio)
            # logits = output.logits
            # logits = logits.detach()
            # f1, acc = bert_accuracy(logits, b_labels)
            f1, acc = scuff_accuracy(output, b_labels)
            total_eval_accuracy += acc
            total_eval_f1 += f1
        avg_val_accuracy = total_eval_accuracy / len(test_loader)
        avg_val_f1 = total_eval_f1 / len(test_loader)
        print("epoch: " + str(epoch) + "\nacc: " + str(avg_val_accuracy) + "\nf1: " + str(avg_val_f1) + "\n")


if __name__ == "__main__":
    main()
    # train_bert()
