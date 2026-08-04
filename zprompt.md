Step 1: Understanding FineWeb-Edu

The dataset you linked is not one giant text file.

It is a collection of Parquet files, where each row is a document with metadata. The default configuration contains about 1.53 billion documents, and Hugging Face also provides smaller sampled configurations (sample-10BT, sample-100BT, sample-350BT) in addition to individual Common Crawl dumps.

Each row looks approximately like:

{
    text: "...actual document...",
    id: "...",
    url: "...",
    date: "...",
    dump: "...",
    language: "en",
    score: 4.78,
    token_count: 932,
    ...
}

Notice that only one field matters for training:

text

Everything else is metadata.

Step 2: Should we use default?

The answer is No.

Not because it's bad.

Because it's too big.

The complete FineWeb-Edu dataset contains around 1.3 trillion tokens of educational web text, which is far beyond what a single RTX 3060 project can realistically process.

Your target:

1–2 Billion tokens

So downloading everything makes no sense.

Step 3: Should we use sample-10BT?

Hugging Face provides:

sample-10BT
sample-100BT
sample-350BT

These are random subsets sampled from the full dataset.

For us:

sample-10BT

contains roughly

10 Billion GPT2 tokens

We only need

1–2 Billion

So this sample is already 5× larger than our needs.

That is perfect.

My Recommendation

Download

sample-10BT

Not

default
Why?

Suppose later you decide

train longer

or

train 5B tokens

or

continue pretraining

You already have more data.

No need to redownload.

Step 4: Never Load the Entire Dataset

Many beginners do

load_dataset(...)

dataset = list(dataset)

Never do this.

Instead

Disk

↓

Streaming

↓

Tokenizer

↓

Binary writer
Step 5: Our Pipeline

I would build this.

FineWeb-Edu

↓

HF Streaming

↓

Document Filter

↓

Tokenizer

↓

Packing

↓

Binary Shards

↓

Training

Notice

Training never reads Parquet.

Why?

Parquet is optimized for

Analytics

not

GPU training

Reading Parquet every batch wastes CPU.

Instead

We'll convert everything once.

Step 6: Binary Dataset

Modern LLMs don't train directly on Hugging Face datasets.

They preprocess into binary.

Like

FineWeb

↓

Tokenizer

↓

uint32 Tokens

↓

train_000.bin

train_001.bin

train_002.bin

...

Then training simply does

mmap()

↓

Read bytes

↓

GPU

Extremely fast.

Why Binary?

Instead of reading

"This is an example sentence..."

the GPU receives

[124,
583,
991,
23,
...
]

Nothing else.

No parsing.

No JSON.

No strings.

Step 7: Packing

This is another thing many tutorials skip.

Suppose

Document 1

200 tokens

Document 2

400 tokens

Document 3

180 tokens

Context

4096

If you train separately

200

400

180

You're wasting

3316 tokens

every step.

Instead

We pack.

Doc1

+

Doc2

+

Doc3

+

...

↓

4096 tokens exactly

This increases GPU utilization significantly.

Step 8: EOS Tokens

Between documents

Document A

<EOS>

Document B

<EOS>

Document C

Never merge documents without EOS.

Otherwise

The model learns

Wikipedia

↓

StackOverflow

↓

Book


as one sentence.

Step 9: Sharding

Suppose

2B tokens

Don't make

train.bin

Make

train_000.bin

train_001.bin

...

train_255.bin

Each

1-2 GB

Advantages

Parallel loading
Easy resume
Easier corruption recovery
Step 10: Data Type

Vocabulary

50000

Needs

16 bits?

No.

Maximum

65535

fits inside uint16.

So

Each token

uint16

instead of

int32

Immediately halves dataset size.

For 2B tokens:

uint16 ≈ 4 GB
uint32 ≈ 8 GB

There is one caveat: this optimization is only safe if your tokenizer vocabulary remains below 65,536 tokens. Since we're planning a 50K vocabulary, uint16 is a good choice.

Step 11: Memory Mapping

Instead of

tokens=np.load(...)

We'll use

numpy.memmap

or

torch.from_file()

Advantages

No RAM usage

OS paging

Lazy loading

Very fast

Even if

Dataset

4GB

RAM usage

~100MB
Step 12: Dataloader

I would NOT use Hugging Face DataLoader.

We'll build our own.

Random shard

↓

Random offset

↓

Read 4097 tokens

↓

Input

↓

Target

Example

Input

1 4 6 8 9

Target

4 6 8 9 2

Exactly like GPT.

Step 13: Validation

Create validation before tokenization.

Example

Training

99.8%

Validation

0.2%

Don't randomly sample every epoch.

Validation must remain fixed.

Step 14: Final Dataset Layout

I would store it like this:

datasets/

└── fineweb_edu/

    raw/
        parquet/

    tokenizer/
        tokenizer.model
        tokenizer.vocab

    processed/

        train/

            shard_000.bin
            shard_001.bin
            ...
            metadata.json

        val/

            shard_000.bin
            metadata.json
metadata.json

This file is incredibly useful.


}