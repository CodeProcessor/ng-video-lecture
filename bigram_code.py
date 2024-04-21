from pathlib import Path

import torch
from loguru import logger

from bigram_model_v2 import BigramLanguageModel
from load_config import ConfigClass

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'model-a2v2.pt'

config = ConfigClass()


"""Read all the input text"""
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
logger.info(f"vocab size: {vocab_size}")
config.set_config('VOCAB_SIZE', vocab_size)

logger.info(f"Chars - {chars}")

# create a mapping from chars
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}

# Encode and decode functions
encode = lambda s: [ctoi[c] for c in s]
decode = lambda a: ''.join([itoc[p] for p in a])

logger.info(decode(encode("Hello world!")))

# split to train and test
data = torch.tensor(encode(text))
p_90 = int(0.9 * len(data))
train_data = data[:p_90]
val_data = data[p_90:]


# get a batch for training
def get_batch(batch_size, split='train'):
    selected_data = train_data if split == 'train' else val_data
    random_pos = torch.randint(0, len(selected_data) - config.config.BLOCK_SIZE, (batch_size,))
    x = torch.stack([selected_data[rp:rp + config.config.BLOCK_SIZE] for rp in random_pos])
    y = torch.stack([selected_data[rp + 1:rp + config.config.BLOCK_SIZE + 1] for rp in random_pos])
    return x.to(DEVICE), y.to(DEVICE)  # (B, T)


# logger.info(get_batch(16))


@torch.no_grad
def evaluate(trained_model):
    output = dict()
    trained_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.config.EVAL_ITERS)
        for i in range(config.config.EVAL_ITERS):
            xb, yb = get_batch(config.config.BATCH_SIZE, split)
            _, loss = trained_model(xb, yb)
            losses[i] = loss.item()
        output[split] = losses.mean()
    trained_model.train()
    return output


if __name__ == '__main__':

    LOAD_MODEL = True
    LEARNING_RATE = float(config.config.LEARNING_RATE)
    model = BigramLanguageModel().to(DEVICE)

    if LOAD_MODEL and Path(MODEL_NAME).is_file():
        state = torch.load(MODEL_NAME)
        model.load_state_dict(state['model_state'])
        del state['model_state']
        logger.info(state)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        for iteration in range(config.config.MAX_ITERATIONS):
            # sample a batch of data
            xb, yb = get_batch(batch_size=config.config.BATCH_SIZE, split='train')

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if iteration % config.config.EVAL_INTERVAL == 0:
                output = evaluate(model)
                logger.info(f"Iteration: {iteration} : training loss : {output['train']:.4f}, val loss : {output['val']:.4f}")

        output = evaluate(model)
        state = {
            "model_state": model.state_dict(),
            "losses": {
                "train":f"{output['train']:.4f}",
                "val": f"{output['val']:.4f}"
            },
            "max_iterations": config.config.MAX_ITERATIONS,
            "lr": config.config.LEARNING_RATE
        }
        torch.save(state, MODEL_NAME)

    # context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    context = torch.tensor([[13]], dtype=torch.long, device=DEVICE)
    logger.info(f"Generated text: {decode(model.generate(context, max_tokens=100)[0].tolist())}")
