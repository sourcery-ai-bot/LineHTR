# Line-level Handwritten Text Recognition with TensorFlow

![poster](https://i.imgur.com/vt0bTYr.png)

This model is an extended version of the [Simple HTR](https://github.com/lamhoangtung/LineHTR) system implemented by [**@Harald Scheidl**](https://github.com/githubharald) and can handle a full line of text image. Huge thanks to [**@Harald Scheidl**](https://github.com/githubharald) for his great works.

# How to run
Go to the `src/` directory and run `python main.py` with these following arguments

## Command line arguments

* `--train`: train the NN, details see below.
* `--validate`: validate the NN, details see below.
* `--beamsearch`: use vanilla beam search decoding (better, but slower) instead of best path decoding.
* `--wordbeamsearch`: use word beam search decoding (only outputs words contained in a dictionary) instead of best path decoding. This is a custom TF operation and must be compiled from source, more information see corresponding section below. It should **not** be used when training the NN.

***I don't include any pretrained model in this branch so you will need to train the model on your data first***

## Train model 

I created this model for the `Cinnamon AI Marathon 2018` competition, they released a small dataset but it's in Vietnamese, so you guys may want to try some other dataset like \[4\]IAM for English.

As long as your dataset contain a `labels.json` file like this:

```
{
    "img1.jpg": "abc xyz",
    ...
    "imgn.jpg": "def ghi"
}
```

With eachkey is the path to the images file and each value is the ground truth label for that image, this code will works fine.

Learning is visualized by Tensorboard, I tracked the character error rate, word error rate and sentences accuracy for this model. All logs will be saved in `./logs/` folder. You can start a Tensorboard session to see the logs with this command `tensorboard --logdir='./logs/'`

It's took me about 48 hours with about 13k images on a single GTX 1060 6GB to get down to 0.16 CER on the private testset of the competition.

## Information about model

### Overview

The model is a extended version of the Simple HTR system [**@Harald Scheidl**](https://github.com/githubharald) implemented
It consists of 7 CNN layers, 2 RNN (Bi-LSTM) layers and the CTC loss and decoding layer and can handle a full line of text image
* The input image is a gray-value image and has a size of 800x64
* 7 CNN layers map the input image to a feature sequence of size 100x512
* 2 LSTM layers with 512 units propagate information through the sequence and map the sequence to a matrix of size 100x205. Each matrix-element represents a score for one of the 205 characters at one of the 100 time-steps
* The CTC layer either calculates the loss value given the matrix and the ground-truth text (when training), or it decodes the matrix to the final text with best path decoding or beam search decoding (when inferring)
* Batch size is set to 50

Highest accuracy achieved is **0.84** on the private testset of the `Cinnamon AI Marathon 2018` competition (measure by Charater Error Rate - CER).


### Improve accuracy

If you need a better accuracy, here are some ideas how to improve it \[2\]:

* Data augmentation: increase dataset-size by applying further (random) transformations to the input images. At the moment, only random distortions are performed.
* Remove cursive writing style in the input images (see [DeslantImg](https://github.com/githubharald/DeslantImg)).
* Increase input size.
* Add more CNN layers or use transfer learning on CNN.
* Replace Bi-LSTM by 2D-LSTM.
* Replace optimizer: Adam improves the accuracy, however, the number of training epochs increases ([see discussion](https://github.com/githubharald/SimpleHTR/issues/27)).
* Decoder: use token passing or word beam search decoding \[3\] (see [CTCWordBeamSearch](https://github.com/githubharald/CTCWordBeamSearch)) to constrain the output to dictionary words.
* Text correction: if the recognized word is not contained in a dictionary, search for the most similar one.

Btw, don't hesitate to ask me anything via a `Github Issue` (See the [issue template file](ISSUE_TEMPLATE.md) for more details)


## References

\[1\] [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5)

\[2\] [Scheidl - Handwritten Text Recognition in Historical Documents](https://repositum.tuwien.ac.at/obvutwhs/download/pdf/2874742)

\[3\] [Scheidl - Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm](https://repositum.tuwien.ac.at/obvutwoa/download/pdf/2774578)

\[4\] [Marti - The IAM-database: an English sentence database for offline handwriting recognition](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
