import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import dice_metric

def train_fn(
    train_loader, 
    val_loader, 
    model, 
    optimizer, 
    loss_fn, 
    num_epoch, 
    device,
    model_save_path,
    writer,
    val_interval=1,
    checkpoint=None,
):
    '''
    Training function

    Params:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        model: Model to be trained
        optimizer: Optimizer for training
        loss_fn: Loss function for training optimization
        num_epoch: Number of epochs for training
        device: Device used for training
        model_save_path: Destination path for best model and model checkpoint saving
        writer: Tensorboard writer
        val_interval: Epoch interval for validation data
        checkpoint: See model_save_path
    '''
    best_metric = -1
    best_metric_epoch = -1
    train_loss_values = []
    val_loss_values = []
    train_metric_values = []
    val_metric_values = []
    continuation_epoch = 0

    if checkpoint is not None:
        print('Loading checkpoint...')
        checkpoint_path = os.path.join(checkpoint, 'checkpoint')

        checkpoint_file = os.listdir(checkpoint_path)[-1]
        print('{}...'.format(checkpoint_file))
        checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_file), map_location='cpu')
        checkpoint_state_dict = checkpoint['model_state_dict']
        optim_state_dict = checkpoint['optim_state_dict']
        model.load_state_dict(checkpoint_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        continuation_epoch = checkpoint['epoch']
        print('Complete')
    
    for epoch in range(num_epoch):
        epoch = (epoch+1) + continuation_epoch
        sum_time = 0
        print('=' * 30,
            '\nEpoch {}/{}'.format(epoch, continuation_epoch + num_epoch)
        )

        epoch_train_loss = 0
        epoch_train_metric = 0
        for train_step, train_data in enumerate(train_loader):
            start = time.time()
            inputs = train_data['img']
            labels = train_data['seg']

            # Add channel to label
            labels_2 = 1 - labels
            labels = torch.cat((labels, labels_2), axis=1)


            inputs, labels = (inputs.to(device), labels.to(device))

            if (train_step+1) % 10 == 0:
                # Write ML
                writer.add_image(
                    'Epoch {} Inputs ML'.format(epoch), 
                    inputs[0, 0, int(inputs.shape[2]/2), :, :],
                    global_step=train_step+1,
                    dataformats='HW',
                )
                writer.add_image(
                    'Epoch {} Labels ML'.format(epoch), 
                    labels[0, 0, int(labels.shape[2]/2), :, :], 
                    global_step=train_step+1,
                    dataformats='HW',
                )

            # Zero optimizer
            optimizer.zero_grad()

            # Forward
            predictions = model(inputs)
            predictions = F.softmax(predictions)
            # predictions = torch.round(predictions)
            if (train_step+1) % 10 == 0:
                writer.add_image(
                    'Epoch {} Outputs ML'.format(epoch), 
                    predictions[0, 0, int(predictions.shape[2]/2), :, :], 
                    global_step=train_step+1,
                    dataformats='HW',
                )

            train_loss = loss_fn(predictions, labels)

            # Backwards
            train_loss.backward()

            # Update optimizer
            optimizer.step()

            # Train loss
            epoch_train_loss += train_loss.item()
            if (train_step+1) % 10 == 0:
                print(
                    f'{train_step+1}/{len(train_loader)}, '
                    f'Train Loss: {train_loss.item():.4f}'
                )

            # Metric loss
            train_metric = dice_metric(predictions, labels)
            epoch_train_metric += train_metric
            if (train_step+1) % 10 == 0:
                print('Train Dice Coefficient: {:.4f}'.format(train_metric))

            end = time.time()
            sum_time += end - start
            if (train_step+1) % 10 == 0:
                print('Epoch train time: {:.1f} min remaining'.format((len(train_loader) * sum_time/(train_step+1)) / 60))

        print('-' * 30)

        epoch_train_loss /= (train_step+1)
        writer.add_scalar('Train Loss per Epoch', epoch_train_loss, epoch)
        print('Epoch loss: {:.4f}'.format(epoch_train_loss))
        train_loss_values.append(epoch_train_loss)
        if not os.path.exists(os.path.join(model_save_path, 'loss_values')):
            os.makedirs(os.path.join(model_save_path, 'loss_values'))
        np.savetxt(
            os.path.join(model_save_path, 'loss_values', 'epoch_{}_train_loss.csv'.format(epoch)), 
            np.asarray(train_loss_values), 
            delimiter=','
        )


        epoch_train_metric /= (train_step+1)
        writer.add_scalar('Dice Coefficient per Epoch', epoch_train_metric, epoch)
        print('Epoch metric: {:.4f}'.format(epoch_train_metric))
        train_metric_values.append(epoch_train_metric)
        np.savetxt(
            os.path.join(model_save_path, 'loss_values', 'epoch_{}_train_metric.csv'.format(epoch)), 
            np.asarray(train_metric_values), 
            delimiter=','
        )


        if (epoch) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_metric = 0

                for val_step, val_data in enumerate(val_loader):

                    val_inputs = val_data['img']
                    val_labels = val_data['seg']
                    # Add channel to label
                    val_labels_2 = 1 - val_labels
                    val_labels = torch.cat((val_labels, val_labels_2), axis=1)
                    
                    val_inputs, val_labels = (val_inputs.to(device), val_labels.to(device))
                    # Write labels
                    writer.add_image(
                        'Epoch {} Val Labels ML'.format(epoch), 
                        val_labels[0, 0, int(val_labels.shape[2]/2), :, :], 
                        global_step=val_step+1,
                        dataformats='HW',
                    )

                    val_predictions = model(val_inputs)
                    val_predictions = F.softmax(val_predictions)
                    # Write preds
                    writer.add_image(
                        'Epoch {} Val Outputs ML'.format(epoch), 
                        val_predictions[0, 0, int(val_predictions.shape[2]/2), :, :], 
                        global_step=train_step+1,
                        dataformats='HW',
                    )
                    val_loss = loss_fn(val_predictions, val_labels)

                    # Val loss
                    epoch_val_loss += val_loss.item()
                    
                    # Metric loss
                    val_metric = dice_metric(val_predictions, val_labels)
                    epoch_val_metric += val_metric
                    

                print('*' * 30)
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation Loss', epoch_val_loss, epoch)
                print('Validation Loss: {:.4f}'.format(epoch_val_loss))
                val_loss_values.append(epoch_val_loss)
                np.savetxt(
                    os.path.join(model_save_path, 'loss_values', 'epoch_{}_val_loss.csv'.format(epoch)), 
                    np.asarray(val_loss_values), 
                    delimiter=','
                )

                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation Dice Coefficient', epoch_val_metric, epoch)
                print('Validation Metric: {:.4f}'.format(epoch_val_metric))
                val_metric_values.append(epoch_val_metric)
                np.savetxt(
                    os.path.join(model_save_path, 'loss_values', 'epoch_{}_val_metric.csv'.format(epoch)), 
                    np.asarray(val_metric_values), 
                    delimiter=','
                )

                # Save checkpoint
                if not os.path.exists(os.path.join(model_save_path, 'checkpoint')):
                    os.makedirs(os.path.join(model_save_path, 'checkpoint'))
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optim_state_dict': optimizer.state_dict(),
                            'train_loss': epoch_train_loss,
                            'val_loss': epoch_val_loss},
                           os.path.join(model_save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch))
                           )
                # if epoch > 0:
                #     os.remove(os.path.join(model_save_path, 'checkpoint/', 'checkpoint_{}.pth'.format(epoch)))

                # Save best model
                if not os.path.exists(os.path.join(model_save_path, 'best')):
                    os.makedirs(os.path.join(model_save_path, 'best'))
                if epoch_val_metric > best_metric:
                    best_metric = epoch_val_metric
                    best_metric_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'best', 'best_model.pth'))

                print(
                    'Current epoch: {} \nCurrent metric: {:.4f} \nBest Metric: {:.4f} at epoch {}'.format(epoch+1, val_metric, best_metric, best_metric_epoch)
                )

    print('Training Completed \n Best mean dice: {:.4f} at epoch {}'.format(best_metric, best_metric_epoch))
    writer.close()
