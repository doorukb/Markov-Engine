# Markov Engine

[![CI](https://github.com/doorukb/Markov-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/doorukb/Markov-Engine/actions/workflows/ci.yml)

- Python 3.10+
- NumPy
- Matplotlib

<br>

A text generation engine built on stochastic matrix mechanics, implemented from scratch using NumPy methods and functions. The project models token transitions as a proper probability matrix rather than a lookup table, which makes the underlying Markov structure visible and mathematically inspectable rather than hidden inside a dict.

The motivating question is not "can this generate useful text" but "what does statistical sequence modeling actually capture, and where does it structurally fail." The engine is built to answer that question concretely. Particularly for outputs of order 1, we can make an analysis of relationship between each word.

<br>

## How it works

The input text is tokenized into a flat sequence of indices. For order-1 models, a square stochastic transition matrix is built directly as a NumPy array where entry `T[i, j]` is the probability of token `j` following token `i`. For order-n models, the state space becomes n-gram tuples and the transition structure is stored as a sparse dict of normalized probability vectors. Generation works by sampling from the appropriate row at each step using `numpy.random.choice` with probability weights. When a higher-order chain reaches a state it never observed (a dead end such as the final n-gram of the corpus), it backs off to progressively shorter contexts ("stupid backoff") rather than emitting uniform-random noise, so one dead end can no longer derail the rest of the output.

The matrix representation is not the most memory-efficient approach for large vocabularies, and it is not meant to be. The point is that the matrix is a real mathematical object you can operate on directly, unlike a Python dictionary.

<br>

## Order and input text size

Higher order means more context and more coherent output, but it also means the chain is more likely to reproduce the source text verbatim. For a small input, the crossover happens quickly. 
For example, order 1 will be more generative and unique compared to the original output, yet it can be gibberish. For order 10, it will be less unique compared to the input, yet more coherent and meaningful.

The vocabulary size directly determines matrix dimensions. For order-1, the matrix is `vocab_size x vocab_size`. 

<br>

## Usage

Point the engine at any plain-text file; a small sample corpus ships in `examples/sample_input.txt`.

```bash
# You want an output of order 1 with maximum size of 150 tokens, and save figures to outputs/ instead of displaying
python main.py examples/sample_input.txt --order 1 --max-tokens 150 --save-figures
```

```bash
# You want an output of order 2 with maximum size of 150 tokens, and display the result in terminal & save as TXT
python main.py examples/sample_input.txt --order 2 --max-tokens 150
```

<br>

## Analysis layer (Only for outputs of order 1)

Available for order-1 models only, where the full stochastic matrix exists as a NumPy array.

**Stationary distribution** finds the left eigenvector of the transition matrix corresponding to eigenvalue 1. This is the long-run probability of being at each token if the chain runs indefinitely.

**Convergence series** computes the state distribution after each of n multiplication steps starting from a uniform prior, showing the chain losing memory of its starting point.

**Entropy per state** computes the Shannon entropy of each row in the transition matrix. High-entropy rows correspond to tokens with uncertain futures. Low-entropy rows are tokens that almost always lead somewhere specific.

**Top transitions** returns the k most probable next tokens for any given input token, with their probabilities.

<br>

## Example Input, Output, Analysis

### Input (170 tokens) : 

And the word of the Lord came unto the man, saying, Arise, and take thee a sandwich of peanut butter, even that which is pressed between two slices of fine bread, and thrust it deep into the narrow slot of the machine of silver, which men call the player of DVDs, that it may be a sign unto this generation. Then the man was sore afraid and cried out, saying, Alas, O Lord, for the bread is wedged tight within the iron jaws of the device, and the creamy substance thereof clingeth to the delicate lasers within, so that it will by no means come forth. But the Lord rebuked him, saying, Am I not the Lord who created both the wheat of the field and the plastics of the earth? Stretch forth thine hand and bring it out. So the man did what the Lord commanded him, and he took a knife of butter, and a wire of copper, and labored with great trembling until the sandwich was delivered whole from the machine, and he bowed his head and gave thanks unto the Most High.


### Output (Order 1, 200 max tokens) : 

for the player of the plastics of the wheat of the word of dvds that it will by no means come forth thine hand and labored with great trembling until the lord rebuked him and gave thanks unto the creamy substance thereof clingeth to the lord came unto the lord commanded him and thrust it out. so that it may be a sandwich of the device and he took a sandwich was sore afraid and he bowed his head and gave thanks unto the wheat of silver which is pressed between two slices of the most high. but the lord came unto the man did what the field and cried out saying arise and cried out. so the lord who created both the machine and he took a sandwich of copper and cried out saying am i not the lord commanded him saying alas o lord for the machine of peanut butter and a sandwich of silver which men call the device and take thee a sign unto this generation. but the bread and the player of copper and the bread and bring it may be a knife of peanut butter even that it deep into


### Analysis : 
#### Convergence Towards Stationary Distribution : 
<img width="1000" height="500" alt="Convergence Toward Stationary Distribution" src="https://github.com/user-attachments/assets/360d5ce0-6951-47d8-b044-08a37bfac214" />

Starting from a uniform distribution over all tokens, the state vector is repeatedly multiplied by the transition matrix. After enough steps it converges to the stationary distribution regardless of the starting point, which is one of the properties of Markov chains. The chain gradually forgets where it began. The plot tracks the probability mass on the five highest-stationary-weight tokens across steps, showing how each one stabilizes as the number of multiplications grows.

#### Entropy per State : 
<img width="1400" height="510" alt="Enthropy" src="https://github.com/user-attachments/assets/25e2fab5-4678-4af9-92f4-9e456eda8cce" />

Measuring the uncertainty in each row of the transition matrix. A row with all its probability mass on one token has entropy zero, as such token has only one plausible successor. A row with uniform probability across all tokens has maximum entropy, and no information about what comes next. The left chart shows the highest-entropy tokens, the ones whose futures are most uncertain. The right chart shows the lowest-entropy tokens, the ones that lead somewhere specific almost every time. This helps us to see which words are pivot points and which are nearly deterministic.

#### Transition Probability Heatmap : 
<img width="1000" height="700" alt="Transition probabilities" src="https://github.com/user-attachments/assets/534979c3-a2ca-44b1-a390-51d5f6360eba" />

Each cell at position (i, j) represents the probability of token j following token i in the input text. Rows are the current token, columns are the next token, and each row sums to 1 by construction. The dense bright cells along a row indicate a token with strong directional preference, and it almost always leads somewhere specific. Sparse or evenly lit rows indicate tokens with many plausible continuations. The heatmap makes the learned transition structure of the corpus directly visible as a geometric object, which no dict-based implementation can produce.

#### Stationary Distribution : 
<img width="1000" height="500" alt="Stationary Distribution" src="https://github.com/user-attachments/assets/7aa02e15-e785-40e0-87d0-d0e400024826" />

The stationary distribution is the long-run probability of the chain being at each token, regardless of where it started. It is computed as the left eigenvector of the transition matrix corresponding to eigenvalue 1. Tokens with high stationary probability are the ones the chain gravitates toward over time, such that they appear frequently not just in the input text but in the probabilistic structure of the transitions themselves. The distribution tells you what the chain "thinks" the text is fundamentally about.
