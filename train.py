import torch
from mixer import MLPMixer
from utils import get_data
from utils import WarmupLinearSchedule, WarmupCosineSchedule
import os
from tqdm import tqdm
import numpy as np
import urllib.request

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(model, output_dir, name):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(output_dir, "%s_checkpoint.bin" % name)
    torch.save(model_to_save.state_dict(), model_checkpoint)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10

    model = MLPMixer(config, args.img_size, num_classes=num_classes, patch_size=16, zero_head=True)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000



def valid(model, test_loader, global_step, device):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits.view(-1, model.n_classes), y.view(-1))
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    return accuracy


def train(model, data_dir, train_batch_size,eval_batch_size, gradient_accumulation_steps, 
          max_grad_norm, learning_rate, warmup_steps, num_steps, 
          weight_decay, decay_type,  device, eval_every):
    
    """ Train the model """
    train_batch_size = train_batch_size // gradient_accumulation_steps

    # Get train and test data
    train_loader, test_loader = get_data(data_dir, img_size, train_batch_size, eval_batch_size)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    t_total = num_steps
    if decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)


    # Training loop
    model.zero_grad()
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if  gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            else:
                loss.backward()

            if (step + 1) %  gradient_accumulation_steps == 0:
                losses.update(loss.item()*gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if global_step % eval_every == 0:
                    accuracy = valid(model, test_loader, global_step, device)
                    if best_acc < accuracy:
                        save_model(model, output_dir='.', name='Mixer-B_16')
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break
            
            
if __name__ == "__main__":
    
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    
    url = 'https://storage.googleapis.com/mixer_models/imagenet21k/Mixer-B_16.npz'
    filename = 'Mixer-B_16.npz'
   # urllib.request.urlretrieve(url, filename)

    img_size = 224
    patch_size = 16
    hidden_dim = 768
    n_blocks = 12
    tokens_mlp_dim = 384
    channels_mlp_dim = 3072
    n_classes=10


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Model & Tokenizer Setup
    model = MLPMixer(img_size=img_size, 
                     patch_size=patch_size, 
                     hidden_dim=hidden_dim,
                     channels_mlp_dim=channels_mlp_dim, 
                     tokens_mlp_dim=tokens_mlp_dim, 
                     n_classes=n_classes, 
                     n_blocks=n_blocks)

    #model.load_from(np.load(args.pretrained_dir))
    model.to(device)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params} M")


    # Training setup
    train_batch_size = 16 # Total batch size for training
    eval_batch_size = 16
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0 
    learning_rate = 3e-2
    warmup_steps = 500
    num_steps = 10000
    weight_decay = 0
    decay_type = "cosine"
    eval_every = 100

    # Training
    train(model, 
          data_dir=data_dir,
          train_batch_size=train_batch_size, 
          eval_batch_size = eval_batch_size,
          gradient_accumulation_steps=gradient_accumulation_steps, 
          max_grad_norm=max_grad_norm, learning_rate=learning_rate, 
          warmup_steps=warmup_steps, 
          num_steps=num_steps, 
          weight_decay=weight_decay, 
          decay_type=decay_type, 
          eval_every=eval_every, 
          device=device)
