from torch.optim import Adam,SGD
from model import *
from data_preparation import *
import torch




def train_and_build(n_epoches):
    for epoch in range(n_epoches):
        print(epoch)
        cnn_model.train()
        for i, (images, labels) in enumerate(train_dataset_loader):
            print("BATCH NUMBER :",i)
            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        cnn_model.eval()
        val_acc_count = 0
        for k, (val_images, val_labels) in enumerate(val_dataset_loader):
            val_outputs = cnn_model(val_images)
            alpha, prediction = torch.max(val_outputs.data, 1)
            # print(test_outputs,'\n',test_outputs.data,'\n',alpha, prediction,'\n', test_labels.data,'\n',"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            val_acc_count += torch.sum(prediction == val_labels.data).item()
            # print(test_acc_count)

        # print(test_acc_count,len(test_total_dataset))
        val_accuracy = val_acc_count / len(val_total_dataset)
        print('For epoch number {} Accuracy {}'.format(epoch,val_accuracy))

if __name__ == '__main__':
    cnn_model = RiceClassifier()
    optimizer = SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    train_and_build(200)
    cnn_model.eval()
    test_acc_count = 0
    for k, (test_images, test_labels) in enumerate(test_dataset_loader):
        test_outputs = cnn_model(test_images)
        alpha, prediction = torch.max(test_outputs.data, 1)
        # print(test_outputs,'\n',test_outputs.data,'\n',alpha, prediction,'\n', test_labels.data,'\n',"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        test_acc_count += torch.sum(prediction == test_labels.data).item()
        # print(test_acc_count)

    # print(test_acc_count,len(test_total_dataset))
    test_accuracy = test_acc_count / len(test_total_dataset)
    print('Accuracy is {}'.format( test_accuracy))

