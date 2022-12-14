import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from monai.metrics import DiceMetric

from utils.utils import dice_metric

def train_fn(
    train_loader, 
    val_loader, 
    model, 
    optimizer, 
    loss_fn, 
    dice_metric,
    num_epoch, 
    device,
    model_save_path,
    writer,
    val_interval=1,
    checkpoint=None,
):
    '''
    Training function with validation intervals 
    :params:
        train_loader:
        val_loader:
        model:
        optimizer:
        loss_fn:
        num_epoch:
        device:
        model_save_path:
        val_interval:
        checkpoint: model_save_path
        writer:
    :outputs:
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

        checkpoint_file = os.listdir(checkpoint_path)[0]
        checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_file), map_location='cpu')
        checkpoint_state_dict = checkpoint['model_state_dict']
        optim_state_dict = checkpoint['optim_state_dict']
        model.load_state_dict(checkpoint_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        continuation_epoch = checkpoint['epoch']
        print('Complete')
    
    for epoch in range(num_epoch):
        sum_time = 0
        print('=' * 30,
            '\nEpoch {}/{}'.format(continuation_epoch + epoch+1, continuation_epoch + num_epoch)
        )

        epoch_train_loss = 0
        epoch_train_metric = 0
        for train_step, train_data in enumerate(train_loader):
            start = time.time()
            inputs = train_data['img']
            labels = train_data['seg']
            inputs, labels = (inputs.to(device), labels.to(device))

            if (train_step+1) % 10 == 0:
                # # Write SI
                # writer.add_image(
                #     'Epoch {} Inputs SI'.format(epoch+1), 
                #     inputs[0, 0, :, :, 160],
                #     global_step=train_step+1,
                #     dataformats='HW',
                # )
                # writer.add_image(
                #     'Epoch {} Labels SI'.format(epoch+1), 
                #     labels[0, 0, :, :, 160], 
                #     global_step=train_step+1,
                #     dataformats='HW',
                # )
                # Write ML
                writer.add_image(
                    'Epoch {} Inputs ML'.format(epoch+1), 
                    inputs[0, 0, 96, :, :],
                    global_step=train_step+1,
                    dataformats='HW',
                )
                writer.add_image(
                    'Epoch {} Labels ML'.format(epoch+1), 
                    labels[0, 0, 96, :, :], 
                    global_step=train_step+1,
                    dataformats='HW',
                )

            # Zero optimizer
            optimizer.zero_grad()

            # Forward
            predictions = model(inputs)
            if (train_step+1) % 10 == 0:
                # Write preds
                # writer.add_image(
                #     'Epoch {} Outputs SI'.format(epoch+1), 
                #     predictions[0, 0, :, :, 160], 
                #     global_step=train_step+1,
                #     dataformats='HW',
                # )
                writer.add_image(
                    'Epoch {} Outputs ML'.format(epoch+1), 
                    predictions[0, 0, 96, :, :], 
                    global_step=train_step+1,
                    dataformats='HW',
                )

            train_loss =  loss_fn(predictions, labels)

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

            # # Metric loss
            # train_metric = dice_metric(predictions, labels)
            # epoch_train_metric += train_metric
            # if (train_step+1) % 10 == 0:
            #     print('Train Dice Coefficient: {:.4f}'.format(train_metric))

            dice_metric(y_pred=predictions, y=labels)

            end = time.time()
            sum_time += end - start
            if (train_step+1) % 10 == 0:
                print('Epoch train time: {:.1f} min remaining'.format((len(train_loader) * sum_time/(train_step+1)) / 60))

        print('-' * 30)

        epoch_train_loss /= (train_step+1)
        writer.add_scalar('Train Loss per Epoch', epoch_train_loss, epoch+1)
        print('Epoch loss: {:.4f}'.format(epoch_train_loss))
        train_loss_values.append(epoch_train_loss)
        np.savetxt(
            os.path.join(model_save_path, 'loss_values', 'epoch_{}_train_loss.csv'.format(epoch+1)), 
            np.asarray(train_loss_values), 
            delimiter=','
        )


        # epoch_train_metric /= (train_step+1)
        # writer.add_scalar('Dice Coefficient per Epoch', epoch_train_metric, epoch+1)
        # print('Epoch metric: {:.4f}'.format(epoch_train_metric))
        # train_metric_values.append(epoch_train_metric)
        # np.savetxt(
        #     os.path.join(model_save_path, 'loss_values', 'epoch_{}_train_metric.csv'.format(epoch+1)), 
        #     np.asarray(train_metric_values), 
        #     delimiter=','
        # )

        epoch_train_metric = dice_metric.aggregate().item()
        dice_metric.reset()
        writer.add_scalar('Dice Coefficient per Epoch', epoch_train_metric, epoch+1)
        print('Epoch metric: {:.4f}'.format(epoch_train_metric))
        train_metric_values.append(epoch_train_metric)
        np.savetxt(
            os.path.join(model_save_path, 'loss_values', 'epoch_{}_train_metric.csv'.format(epoch+1)), 
            np.asarray(train_metric_values), 
            delimiter=','
        )


        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                epoch_val_loss = 0
                epoch_val_metric = 0

                for val_step, val_data in enumerate(val_loader):

                    val_inputs = val_data['img']
                    val_labels = val_data['seg']
                    val_inputs, val_labels = (val_inputs.to(device), val_labels.to(device))
                    # Write labels
                    # writer.add_image(
                    #     'Epoch {} Val Labels SI'.format(epoch+1), 
                    #     val_labels[0, 0, :, :, 160], 
                    #     global_step=val_step+1,
                    #     dataformats='HW',
                    # )
                    writer.add_image(
                        'Epoch {} Val Labels ML'.format(epoch+1), 
                        val_labels[0, 0, 96, :, :], 
                        global_step=val_step+1,
                        dataformats='HW',
                    )

                    val_predictions = model(val_inputs)
                    # Write preds
                    # writer.add_image(
                    #     'Epoch {} Val Outputs SI'.format(epoch+1), 
                    #     val_predictions[0, 0, :, :, 160], 
                    #     global_step=val_step+1,
                    #     dataformats='HW',
                    # )
                    writer.add_image(
                        'Epoch {} Val Outputs ML'.format(epoch+1), 
                        val_predictions[0, 0, 96, :, :], 
                        global_step=train_step+1,
                        dataformats='HW',
                    )

                    # Val loss
                    val_loss = loss_fn(val_predictions, val_labels)
                    epoch_val_loss += val_loss.item()
                    
                    # Metric loss
                    # val_metric = dice_metric(val_predictions, val_labels)
                    # epoch_val_metric += val_metric
                    dice_metric(y_pred=val_predictions, y=val_labels)

                    

                print('*' * 30)
                epoch_val_loss /= (val_step+1)
                writer.add_scalar('Validation Loss', epoch_val_loss, epoch+1)
                print('Validation Dice Loss: {:.4f}'.format(epoch_val_loss))
                val_loss_values.append(epoch_val_loss)
                np.savetxt(
                    os.path.join(model_save_path, 'loss_values', 'epoch_{}_val_loss.csv'.format(epoch+1)), 
                    np.asarray(val_loss_values), 
                    delimiter=','
                )

                epoch_val_metric /= (val_step+1)
                writer.add_scalar('Validation Dice Coefficient', epoch_val_metric, epoch+1)
                print('Validation Dice Loss: {:.4f}'.format(epoch_val_metric))
                val_metric_values.append(epoch_val_metric)
                np.savetxt(
                    os.path.join(model_save_path, 'loss_values', 'epoch_{}_val_metric.csv'.format(epoch+1)), 
                    np.asarray(val_metric_values), 
                    delimiter=','
                )

                # Save checkpoint
                if not os.path.exists(os.path.join(model_save_path, 'checkpoint')):
                    os.makedirs(os.path.join(model_save_path, 'checkpoint'))
                torch.save({'epoch': epoch+1,
                            'model_state_dict': model.state_dict(),
                            'optim_state_dict': optimizer.state_dict(),
                            'train_loss': epoch_train_loss,
                            'val_loss': epoch_val_loss},
                           os.path.join(model_save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch + 1))
                           )
                if epoch > 0:
                    os.remove(os.path.join(model_save_path, 'checkpoint/', 'checkpoint_{}.pth'.format(epoch)))

                # Save best model
                if not os.path.exists(os.path.join(model_save_path, 'best')):
                    os.makedirs(os.path.join(model_save_path, 'best'))
                if epoch_val_metric > best_metric:
                    best_metric = epoch_val_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'best', 'best_model.pth'))

                print(
                    'Current epoch: {} \nCurrent dice coefficient: {:.4f} \nBest dice coefficient: {:.4f} at epoch {}'.format(epoch+1, epoch_val_metric, best_metric, best_metric_epoch)
                )

    print('Training Completed \n Best mean dice: {:.4f} at epoch {}'.format(best_metric, best_metric_epoch))
    writer.close()
