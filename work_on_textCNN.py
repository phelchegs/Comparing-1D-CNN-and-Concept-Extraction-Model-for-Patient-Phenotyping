import preprocess
import textCNN
import utils
import torch
import argparse
# from torch.optim.lr_scheduler import LambdaLR

def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annot_text_path', type = str, help = 'Path to annot_text.csv')
    parser.add_argument('--w2v_path', type = str, help = 'Path to w2v.txt')
    parser.add_argument('--data_filename', type = str, help = 'Path to the pickle file that saves train, validate, and test data')
    parser.add_argument('--batchsize', type = int, help = 'Batch size for training', default = 1)
    parser.add_argument('--train_size', type = float, help = 'Percentage of training', default = 0.7)
    parser.add_argument('--valid_size', type = float, help = 'Percentage of validation', default = 0.2)
    parser.add_argument('--cnn_channels', type = int, help = 'NO. of channels after CNN treatment', default = 3)
    parser.add_argument('--cnn_windows', type = int, nargs = '+', help = "List of CNNs' different sizes", default = [1, 2, 3])
    parser.add_argument('--epochs', type = int, help = "number of epochs", default = 10)
    parser.add_argument('--target_name', type = str, help = "target (condition) name", default = 'cohort')
    args = parser.parse_args()
    
    conditions = ['cohort', #0
                  'Obesity', #1
                  'Non_Adherence', #2
                  'Developmental_Delay_Retardation', #3
                  'Advanced_Heart_Disease', #4
                  'Advanced_Lung_Disease', #5
                  'Schizophrenia_and_other_Psychiatric_Disorders', #6
                  'Alcohol_Abuse', #7
                  'Other_Substance_Abuse', #8
                  'Chronic_Pain_Fibromyalgia', #9
                  'Chronic_Neurological_Dystrophies', #10
                  'Advanced_Cancer', #11
                  'Depression', #12
                  'Dementia', #13
                  'Unsure'] #14
    
    train_dataloader, val_dataloader, test_dataloader, embeddings, padding_index, weights = preprocess.getdata(args.annot_text_path, args.w2v_path, args.data_filename, args.batchsize, args.train_size, args.valid_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train worker process using {device} for training", flush = True)
    torch.manual_seed(1987)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        
    model = textCNN.textCNN(embeddings, embeddings.shape[1], args.cnn_channels, args.cnn_windows, 2, dropout = 0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    # lr_scheduler = LambdaLR(
    #     optimizer = optimizer,
    #     lr_lambda = lambda step: utils.lr_scheduler(step, model_size = embeddings.shape[1], factor = 1.0, warmup_step = 400))
    

    print('Now working on condition {}.'.format(args.target_name))
    
    if torch.cuda.is_available():
        model.cuda(device)
    train_state = utils.Train_Counter
    best_acc = -1
    for epoch in range(args.epochs):
        print(f"Now Training Epoch {epoch} ====", flush = True)
        loss, token, train_state = textCNN.train_one_epoch(model, (utils.Batchify(b[0], b[1], padding_index, conditions.index(args.target_name)) for b in train_dataloader), device, optimizer, weights[conditions.index(args.target_name)], train_state)
        print('loss {:.2e}, token {:.2e} in epoch {}.'.format(loss, token, epoch))
        torch.cuda.empty_cache()
    
        print(f"Check the validation results for epoch {epoch} ====", flush = True)
        with torch.no_grad():
            val_loss, accuracy = textCNN.val_one_epoch(model, (utils.Batchify(b[0], b[1], padding_index, conditions.index(args.target_name)) for b in val_dataloader), weights[conditions.index(args.target_name)], device)
            print('Validation loss {:.2e}, accuracy {:.2%}.'.format(val_loss, accuracy))
        if accuracy > best_acc:
            best_acc = accuracy
            print('Model saves at accuracy {:.2%}.'.format(best_acc))
            torch.save(model.state_dict(), "textCNN_{}_bestaccu.pt".format(args.target_name))
        torch.cuda.empty_cache()
    
    print('NO. of samples {}, NO. of tokens {} processed after the whole training.'.format(train_state.samples, train_state.tokens))

    
    