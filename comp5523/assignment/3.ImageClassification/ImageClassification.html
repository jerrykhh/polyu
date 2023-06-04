<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="generator" content="pandoc" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
  <meta name="author" content="LI, Qimai; Zhong, Yongfeng" />
  <meta name="dcterms.date" content="November 4, 2021; November 9,
2022" />
  <title>Image Classification via CNN</title>
  <style>
    code{white-space: pre-wrap;}
    span.smallcaps{font-variant: small-caps;}
    div.columns{display: flex; gap: min(4vw, 1.5em);}
    div.column{flex: auto; overflow-x: auto;}
    div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;}
    ul.task-list{list-style: none;}
    ul.task-list li input[type="checkbox"] {
      width: 0.8em;
      margin: 0 0.8em 0.2em -1.6em;
      vertical-align: middle;
    }
    pre > code.sourceCode { white-space: pre; position: relative; }
    pre > code.sourceCode > span { display: inline-block; line-height: 1.25; }
    pre > code.sourceCode > span:empty { height: 1.2em; }
    .sourceCode { overflow: visible; }
    code.sourceCode > span { color: inherit; text-decoration: inherit; }
    div.sourceCode { margin: 1em 0; }
    pre.sourceCode { margin: 0; }
    @media screen {
    div.sourceCode { overflow: auto; }
    }
    @media print {
    pre > code.sourceCode { white-space: pre-wrap; }
    pre > code.sourceCode > span { text-indent: -5em; padding-left: 5em; }
    }
    pre.numberSource code
      { counter-reset: source-line 0; }
    pre.numberSource code > span
      { position: relative; left: -4em; counter-increment: source-line; }
    pre.numberSource code > span > a:first-child::before
      { content: counter(source-line);
        position: relative; left: -1em; text-align: right; vertical-align: baseline;
        border: none; display: inline-block;
        -webkit-touch-callout: none; -webkit-user-select: none;
        -khtml-user-select: none; -moz-user-select: none;
        -ms-user-select: none; user-select: none;
        padding: 0 4px; width: 4em;
        color: #aaaaaa;
      }
    pre.numberSource { margin-left: 3em; border-left: 1px solid #aaaaaa;  padding-left: 4px; }
    div.sourceCode
      {   }
    @media screen {
    pre > code.sourceCode > span > a:first-child::before { text-decoration: underline; }
    }
    code span.al { color: #ff0000; font-weight: bold; } /* Alert */
    code span.an { color: #60a0b0; font-weight: bold; font-style: italic; } /* Annotation */
    code span.at { color: #7d9029; } /* Attribute */
    code span.bn { color: #40a070; } /* BaseN */
    code span.bu { color: #008000; } /* BuiltIn */
    code span.cf { color: #007020; font-weight: bold; } /* ControlFlow */
    code span.ch { color: #4070a0; } /* Char */
    code span.cn { color: #880000; } /* Constant */
    code span.co { color: #60a0b0; font-style: italic; } /* Comment */
    code span.cv { color: #60a0b0; font-weight: bold; font-style: italic; } /* CommentVar */
    code span.do { color: #ba2121; font-style: italic; } /* Documentation */
    code span.dt { color: #902000; } /* DataType */
    code span.dv { color: #40a070; } /* DecVal */
    code span.er { color: #ff0000; font-weight: bold; } /* Error */
    code span.ex { } /* Extension */
    code span.fl { color: #40a070; } /* Float */
    code span.fu { color: #06287e; } /* Function */
    code span.im { color: #008000; font-weight: bold; } /* Import */
    code span.in { color: #60a0b0; font-weight: bold; font-style: italic; } /* Information */
    code span.kw { color: #007020; font-weight: bold; } /* Keyword */
    code span.op { color: #666666; } /* Operator */
    code span.ot { color: #007020; } /* Other */
    code span.pp { color: #bc7a00; } /* Preprocessor */
    code span.sc { color: #4070a0; } /* SpecialChar */
    code span.ss { color: #bb6688; } /* SpecialString */
    code span.st { color: #4070a0; } /* String */
    code span.va { color: #19177c; } /* Variable */
    code span.vs { color: #4070a0; } /* VerbatimString */
    code span.wa { color: #60a0b0; font-weight: bold; font-style: italic; } /* Warning */
  </style>
  <link rel="stylesheet" href="image/github-pandoc.css" />
  <script
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml-full.js"
  type="text/javascript"></script>
  <!--[if lt IE 9]>
    <script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js"></script>
  <![endif]-->
</head>
<body>
<header id="title-block-header">
<h1 class="title">Image Classification via CNN</h1>
</header>
<nav id="TOC" role="doc-toc">
<ul>
<li><a href="#install-pytorch" id="toc-install-pytorch"><span
class="toc-section-number">1</span> Install PyTorch</a>
<ul>
<li><a href="#on-macos" id="toc-on-macos"><span
class="toc-section-number">1.1</span> On macOS</a></li>
<li><a href="#on-windows" id="toc-on-windows"><span
class="toc-section-number">1.2</span> On Windows</a></li>
</ul></li>
<li><a href="#cifar10-dataset" id="toc-cifar10-dataset"><span
class="toc-section-number">2</span> CIFAR10 Dataset</a></li>
<li><a href="#create-cnn-model" id="toc-create-cnn-model"><span
class="toc-section-number">3</span> Create CNN Model</a>
<ul>
<li><a href="#convolutional-layers" id="toc-convolutional-layers"><span
class="toc-section-number">3.1</span> Convolutional layers</a></li>
<li><a href="#activation" id="toc-activation"><span
class="toc-section-number">3.2</span> Activation</a></li>
<li><a href="#pooling-layers-subsampling"
id="toc-pooling-layers-subsampling"><span
class="toc-section-number">3.3</span> Pooling layers
(subsampling)</a></li>
<li><a href="#fully-connected-fc-layers"
id="toc-fully-connected-fc-layers"><span
class="toc-section-number">3.4</span> Fully connected (FC)
layers</a></li>
<li><a href="#create-lenet-5" id="toc-create-lenet-5"><span
class="toc-section-number">3.5</span> Create LeNet-5</a></li>
</ul></li>
<li><a href="#train-the-model" id="toc-train-the-model"><span
class="toc-section-number">4</span> Train the Model</a></li>
<li><a href="#test-the-model" id="toc-test-the-model"><span
class="toc-section-number">5</span> Test the Model</a></li>
<li><a href="#load-predefined-cnns" id="toc-load-predefined-cnns"><span
class="toc-section-number">6</span> Load Predefined CNNs</a></li>
<li><a href="#gpu-acceleration" id="toc-gpu-acceleration"><span
class="toc-section-number">7</span> GPU Acceleration</a>
<ul>
<li><a href="#install-cuda" id="toc-install-cuda"><span
class="toc-section-number">7.1</span> Install CUDA</a></li>
<li><a href="#adapt-your-code-for-gpu"
id="toc-adapt-your-code-for-gpu"><span
class="toc-section-number">7.2</span> Adapt your code for GPU</a></li>
</ul></li>
<li><a href="#assignment" id="toc-assignment"><span
class="toc-section-number">8</span> Assignment</a>
<ul>
<li><a href="#handwritten-digit-recognition-18-points"
id="toc-handwritten-digit-recognition-18-points"><span
class="toc-section-number">8.1</span> Handwritten digit recognition (18
points)</a></li>
<li><a href="#bonus-fashion-mnist-3-points"
id="toc-bonus-fashion-mnist-3-points"><span
class="toc-section-number">8.2</span> Bonus: Fashion-MNIST (3
points)</a></li>
<li><a href="#submission-instruction"
id="toc-submission-instruction"><span
class="toc-section-number">8.3</span> Submission instruction</a></li>
</ul></li>
</ul>
</nav>
<!-- # Image Classification via CNN -->
<p>This tutorial teaches you how to classify images via convolutional
neural network (CNN). You will learn how to create, train and evaluate a
CNN by PyTorch. Part of this tutorial is adapted from <a
href="https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py">Deep
Learning with PyTorch: A 60 Minute Blitz</a>.</p>
<h1 data-number="1" id="install-pytorch"><span
class="header-section-number">1</span> Install PyTorch</h1>
<p>PyTorch is an open source machine learning library, used for
applications such as computer vision and natural language processing,
primarily developed by Facebook’s AI Research lab. Please install it via
following command.</p>
<h2 data-number="1.1" id="on-macos"><span
class="header-section-number">1.1</span> On macOS</h2>
<p>Please open your terminal, and install PyTorch via following
command.</p>
<pre><code>$ conda install pytorch torchvision torchaudio -c pytorch</code></pre>
<p>Then test your installation and you shall see your torch version
number, such as <code>1.10</code>.</p>
<pre><code>$ python -c &quot;import torch; print(torch.__version__)&quot;
1.10.0</code></pre>
<h2 data-number="1.2" id="on-windows"><span
class="header-section-number">1.2</span> On Windows</h2>
<p>Please open your anaconda prompt from Start menu, and install PyTorch
via following command.</p>
<pre><code>(base) C:\Users\%USERNAME%&gt; conda install pytorch torchvision torchaudio cpuonly -c pytorch</code></pre>
<p>Then test your installation and you shall see your torch version
number, such as <code>1.10</code>.</p>
<pre><code>(base) C:\Users\%USERNAME%&gt; python -c &quot;import torch; print(torch.__version__)&quot;
1.10.0</code></pre>
<h1 data-number="2" id="cifar10-dataset"><span
class="header-section-number">2</span> CIFAR10 Dataset</h1>
<p>For this tutorial, we will use the CIFAR10 dataset. The CIFAR-10
dataset consists of 60000 32x32 colour images in 10 classes, with 6000
images per class. There are 50000 training images and 10000 test images.
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images
of 32x32 pixels in size. Note that PyTorch take channels as first
dimension by convention. This convention is different from other
platform such as Pillow, Matlab, skimage, etc. They all put channels at
last dimension.</p>
<center>
<img src="https://pytorch.org/tutorials/_images/cifar10.png">
<p>
Figure 1. CIFAR10 dataset
</p>
</center>
<p>We can load CIFAR10 from torchvision. It may take several minutes to
download the dataset.</p>
<div class="sourceCode" id="cb5"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> torchvision.datasets <span class="im">import</span> CIFAR10</span>
<span id="cb5-2"><a href="#cb5-2" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> torchvision.transforms <span class="im">import</span> ToTensor</span>
<span id="cb5-3"><a href="#cb5-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb5-4"><a href="#cb5-4" aria-hidden="true" tabindex="-1"></a>trainset <span class="op">=</span> CIFAR10(root<span class="op">=</span><span class="st">&#39;./data&#39;</span>, train<span class="op">=</span><span class="va">True</span>,</span>
<span id="cb5-5"><a href="#cb5-5" aria-hidden="true" tabindex="-1"></a>                   download<span class="op">=</span><span class="va">True</span>, transform<span class="op">=</span>ToTensor())</span>
<span id="cb5-6"><a href="#cb5-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb5-7"><a href="#cb5-7" aria-hidden="true" tabindex="-1"></a>testset <span class="op">=</span> CIFAR10(root<span class="op">=</span><span class="st">&#39;./data&#39;</span>, train<span class="op">=</span><span class="va">False</span>,</span>
<span id="cb5-8"><a href="#cb5-8" aria-hidden="true" tabindex="-1"></a>                  download<span class="op">=</span><span class="va">True</span>, transform<span class="op">=</span>ToTensor())</span></code></pre></div>
<p>The dataset consists of two parts. One is train set, another one is
test set. Usually, we train our model (CNN) on train set, and test the
model on test set.</p>
<ul>
<li>Train: Show CNN the images, and tell it which classes they belong
to. In such a way, we can teach it to distinguish different
classes.</li>
<li>Test: Show CNN the images, and ask it which classes they belong to.
In such a way, we can test how well the CNN learns.</li>
</ul>
<p><code>trainset.classes</code> contains the all class names in
order.</p>
<div class="sourceCode" id="cb6"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb6-1"><a href="#cb6-1" aria-hidden="true" tabindex="-1"></a>trainset.classes</span>
<span id="cb6-2"><a href="#cb6-2" aria-hidden="true" tabindex="-1"></a><span class="co"># [&#39;airplane&#39;, &#39;automobile&#39;, &#39;bird&#39;, &#39;cat&#39;, &#39;deer&#39;, &#39;dog&#39;, &#39;frog&#39;, &#39;horse&#39;, &#39;ship&#39;, &#39;truck&#39;]</span></span></code></pre></div>
<p>Train set contains 50000 images. Let’s get the first image in train
set, and show it.</p>
<div class="sourceCode" id="cb7"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a><span class="bu">len</span>(trainset)              <span class="co"># 50000 images</span></span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a>image, label <span class="op">=</span> trainset[<span class="dv">0</span>] <span class="co"># get first image and its class id</span></span>
<span id="cb7-3"><a href="#cb7-3" aria-hidden="true" tabindex="-1"></a>image.shape                <span class="co"># 3 x 32 x 32</span></span>
<span id="cb7-4"><a href="#cb7-4" aria-hidden="true" tabindex="-1"></a>imshow(image)              <span class="co"># `imshow` is in cifar10.py</span></span>
<span id="cb7-5"><a href="#cb7-5" aria-hidden="true" tabindex="-1"></a>trainset.classes[label]    <span class="co"># &#39;frog&#39;</span></span></code></pre></div>
<p>You can see the image is of shape <span
class="math inline">\(3\times32\times32\)</span>, which means it has
<span class="math inline">\(3\)</span> channels, and <span
class="math inline">\(32\times32\)</span> pixels.</p>
<p>Script <code>cifar10.py</code> already contains all code you need to
load the dataset. In your program, all you need to do is</p>
<div class="sourceCode" id="cb8"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb8-1"><a href="#cb8-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> dataset <span class="im">import</span> load_cifar10, imshow</span>
<span id="cb8-2"><a href="#cb8-2" aria-hidden="true" tabindex="-1"></a>trainset, testset <span class="op">=</span> load_cifar10()</span></code></pre></div>
<p>Beside the dataset itself, we also need <code>DataLoader</code>
objects to help us randomly load image batch by batch. Batch means a
small collection of images. Here, we set <code>batch_size</code> to 4,
so each batch contains 4 images.</p>
<div class="sourceCode" id="cb9"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb9-1"><a href="#cb9-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> torch.utils.data <span class="im">import</span> DataLoader</span>
<span id="cb9-2"><a href="#cb9-2" aria-hidden="true" tabindex="-1"></a>trainloader <span class="op">=</span> DataLoader(trainset, batch_size<span class="op">=</span><span class="dv">4</span>, shuffle<span class="op">=</span><span class="va">True</span>)</span>
<span id="cb9-3"><a href="#cb9-3" aria-hidden="true" tabindex="-1"></a>testloader <span class="op">=</span> DataLoader(testset, batch_size<span class="op">=</span><span class="dv">4</span>, shuffle<span class="op">=</span><span class="va">False</span>)</span></code></pre></div>
<p>Then we may iterate over the <code>DataLoader</code>, to get batches
until the dataset is exhausted.</p>
<div class="sourceCode" id="cb10"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb10-1"><a href="#cb10-1" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> batch <span class="kw">in</span> trainloader:</span>
<span id="cb10-2"><a href="#cb10-2" aria-hidden="true" tabindex="-1"></a>    images, labels <span class="op">=</span> batch</span>
<span id="cb10-3"><a href="#cb10-3" aria-hidden="true" tabindex="-1"></a>    <span class="bu">print</span>(images.shape) <span class="co"># [4, 3, 32, 32]</span></span>
<span id="cb10-4"><a href="#cb10-4" aria-hidden="true" tabindex="-1"></a>    <span class="bu">print</span>(labels.shape) <span class="co"># [4]</span></span>
<span id="cb10-5"><a href="#cb10-5" aria-hidden="true" tabindex="-1"></a>    <span class="cf">break</span></span></code></pre></div>
<p><code>images</code> is of shape <code>[4, 3, 32, 32]</code>, which
means it contains 4 images, each has 3 channels, and is of size 32x32.
<code>labels</code> contains 4 scalars, which are the class IDs of this
batch.</p>
<h1 data-number="3" id="create-cnn-model"><span
class="header-section-number">3</span> Create CNN Model</h1>
<p>In this tutorial, we implement a simple but famous CNN model,
LeNet-5. 5 means it contains 5 (convolutional or fully-connected)
layers.</p>
<center>
<img src="https://www.researchgate.net/profile/Vladimir_Golovko3/publication/313808170/figure/fig3/AS:552880910618630@1508828489678/Architecture-of-LeNet-5.png">
<p>
Figure 2. Architecture of LeNet-5
</p>
</center>
<p>A typical CNN consists of these kinds of layer — convolutional
layers, max pooling layers, and fully connected layers.</p>
<h2 data-number="3.1" id="convolutional-layers"><span
class="header-section-number">3.1</span> Convolutional layers</h2>
<p>Convolutional layers are usually the first several layers. They
perform a convolution on the output of last layer to extract features
from the image. A convolutional layer has three architecture
parameters:</p>
<ul>
<li>kernel_size <span class="math inline">\(h\times w\)</span>: the size
of the convolutional kernel.</li>
<li>in_channels: the number of input channels</li>
<li>out_channels: the number of output channels</li>
</ul>
<center>
<img src="https://user-images.githubusercontent.com/62511046/84712471-4cf7ad00-af86-11ea-92a6-ea3cacab3403.png" style="width: 90%">
</center>
<p>In this layer, beside the convolutional kernel <span
class="math inline">\(K\)</span>, we also have a bias <span
class="math inline">\(b\)</span> added to each output channel. The
formula for output is <span class="math display">\[X&#39; = K * X +
b,\]</span> where <span class="math inline">\(*\)</span> stands for
convolution, <span class="math inline">\(X\)</span> and <span
class="math inline">\(X&#39;\)</span> are input and output. The total
number of trainable parameters in convolutional layer is:</p>
<p><span class="math display">\[\underbrace{h\times w \times
\text{in\_channels}\times\text{out\_channels}}_\text{kernel}
+ \underbrace{\text{out\_channels}}_\text{bias}\]</span></p>
<p>The convolution is performed without padding by default, so image
size will shrink after convolution. If input image size is <span
class="math inline">\(H\times W\)</span> and the kernel size is <span
class="math inline">\(h\times w\)</span>, the output will be of size
<span class="math display">\[(H+1-h) \times (W+1-w).\]</span> Then we
take channels and batch size into consideration, assume the input tensor
has shape [batch_size, in_channels, H, W], then the output tensor will
have shape</p>
<ul>
<li>input shape: [batch_size, in_channels, H, W]</li>
<li>output shape: [batch_size, out_channels, H+1-h, W+1-w]</li>
</ul>
<h2 data-number="3.2" id="activation"><span
class="header-section-number">3.2</span> Activation</h2>
<p>The output of convolutional layer and fully connected layer is
usually “activated”, i.e., transformed by a non-linear function, such as
ReLU, sigmoid, tanh, etc. Activation functions are all scalar function.
They do not change the tensor shape, but only map each element into a
new value. They usually contain no trainable parameters.</p>
<p>In this tutorial, we choose perhaps the most popular activation
function, <span class="math inline">\(ReLU(x) = \max(0, x)\)</span>.</p>
<center>
<img src="https://user-images.githubusercontent.com/13168096/49909393-11c47b80-fec2-11e8-8fcd-d9d54b8b0258.png" style="
    width: 500px; /* width of container */
    height: 220px; /* height of container */
    object-fit: cover;
    object-position: 0px -30px; /* try 20px 10px */
    <!-- border: 5px solid black; -->
    ">
<p>
Figure 3. Activation Functions
</p>
</center>
<p>Here we demonstrate how to create the first convolutional layer of
LeNet-5 by PyTorch. This layer has kernel size 5x5 and its output
contains 6 channels. Its input is the original RGB images, so
<code>in_channels=3</code>. The output is activated by ReLU (Original
paper uses tanh).</p>
<div class="sourceCode" id="cb11"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb11-1"><a href="#cb11-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> torch.nn <span class="im">as</span> nn</span>
<span id="cb11-2"><a href="#cb11-2" aria-hidden="true" tabindex="-1"></a><span class="co"># convolutional layer 1</span></span>
<span id="cb11-3"><a href="#cb11-3" aria-hidden="true" tabindex="-1"></a>conv_layer1 <span class="op">=</span> nn.Sequential(</span>
<span id="cb11-4"><a href="#cb11-4" aria-hidden="true" tabindex="-1"></a>    nn.Conv2d(in_channels<span class="op">=</span><span class="dv">3</span>, out_channels<span class="op">=</span><span class="dv">6</span>, kernel_size<span class="op">=</span>(<span class="dv">5</span>,<span class="dv">5</span>)),</span>
<span id="cb11-5"><a href="#cb11-5" aria-hidden="true" tabindex="-1"></a>    nn.ReLU(),</span>
<span id="cb11-6"><a href="#cb11-6" aria-hidden="true" tabindex="-1"></a>)</span></code></pre></div>
<h2 data-number="3.3" id="pooling-layers-subsampling"><span
class="header-section-number">3.3</span> Pooling layers
(subsampling)</h2>
<p>Pooling usually follows a convolutional layer. There are two kinds of
pooling layer — maximum pooling and average pooling. Maximum pooling
computes the maximum of small local patches, while average pooling
computes the average of small local patches.</p>
<center>
<img src="https://computersciencewiki.org/images/8/8a/MaxpoolSample2.png" style="width:50%">
<p>
Figure 4. Max pooling with kernel size 2x2
</p>
</center>
<p>The kernel size of a pooling layer is the size of local patches.
Assume the input image is of size <span class="math inline">\(H\times
W\)</span> and the kernel size is <span class="math inline">\(h\times
w\)</span>, the output of pooling layer will be of size <span
class="math display">\[\frac{H}{h} \times \frac{W}{w}.\]</span> Then we
take channels and batch size into consideration, input tensor and output
tensor will have shape:</p>
<ul>
<li>input shape: [batch_size, in_channels, H, W],</li>
<li>output shape: [batch_size, in_channels, H/h, W/w].</li>
</ul>
<p>Pooling layers do not change the number of channels and do not
contain any trainable parameters.</p>
<p>This code snip demonstrates how to create a 2x2 max pooling
layer.</p>
<div class="sourceCode" id="cb12"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb12-1"><a href="#cb12-1" aria-hidden="true" tabindex="-1"></a>max_pool <span class="op">=</span> nn.MaxPool2d(kernel_size<span class="op">=</span>(<span class="dv">2</span>,<span class="dv">2</span>))</span></code></pre></div>
<h2 data-number="3.4" id="fully-connected-fc-layers"><span
class="header-section-number">3.4</span> Fully connected (FC)
layers</h2>
Fully connected (FC) layers are usually the last several layers. They
take the features conv layers produced and output the final
classification result. Before go into FC layers, we need to “flatten”
the intermediate representation produced by convolutional layers. The
output of CNN is a 4D tensor of shape [batch_size, channels, H, W].
After flattened, it becomes a 2D tensor of shape [batch_size,
channels*H*W]. This 2D tensor is exactly what FC layers consumes as
input.
<center>
<p>4D tensor of shape [batch_size, channels, H, W]</p>
|<br/> flatten<br/> |<br/> v<br/> 2D tensor of shape [batch_size,
channels*H*W]
</center>
<p>A FC layer has two architecture parameter — input features and output
features:</p>
<ul>
<li>in_features: the number of input features,</li>
<li>out_features: the number of output features.</li>
</ul>
<center>
<img src="https://www.researchgate.net/profile/Srikanth_Tammina/publication/337105858/figure/fig3/AS:822947157643267@1573217300519/Types-of-pooling-d-Fully-Connected-Layer-At-the-end-of-a-convolutional-neural-network.jpg" style="width:30%">
<p>
Figure 5. FC layer with 7 input features and 5 output features
</p>
</center>
<p>The input and output of FC layers are of shape:</p>
<ul>
<li>input shape: [batch_size, in_features]</li>
<li>output shape: [batch_size, out_features]</li>
</ul>
<p>The formula for output is <span class="math display">\[X&#39; =
\Theta X + b\]</span> where <span class="math inline">\(\Theta\)</span>
is weights, and <span class="math inline">\(b\)</span> is biases.
Because there is a weight between any input feature and any output
feature, <span class="math inline">\(\Theta\)</span> is of shape <span
class="math inline">\(\text{in\_features} \times
\text{out\_features}.\)</span> Number of biases is equal to the number
of output features. Each output feature is added by a bias. In total,
the number of trainable parameters in a FC layer is <span
class="math display">\[\underbrace{\text{in\_features} \times
\text{out\_features}}_{\text{weights}~\Theta}
+\underbrace{\text{out\_features}}_\text{bias}.\]</span></p>
<p>This example shows how to create a FC layer in PyTorch. The created
FC layer has 120 input features and 84 output features, and its output
is activated by ReLU.</p>
<div class="sourceCode" id="cb13"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb13-1"><a href="#cb13-1" aria-hidden="true" tabindex="-1"></a>fc_layer <span class="op">=</span> nn.Sequential(</span>
<span id="cb13-2"><a href="#cb13-2" aria-hidden="true" tabindex="-1"></a>    nn.Linear(in_features<span class="op">=</span><span class="dv">120</span>, out_features<span class="op">=</span><span class="dv">84</span>),</span>
<span id="cb13-3"><a href="#cb13-3" aria-hidden="true" tabindex="-1"></a>    nn.ReLU(),</span>
<span id="cb13-4"><a href="#cb13-4" aria-hidden="true" tabindex="-1"></a>)</span></code></pre></div>
<p>The last layer of our CNN is a little bit special. First, it is not
activated, i.e., no ReLU. Second, its output features must be equal to
the number of classes. Here, we have 10 classes in total, so its output
features must be 10.</p>
<div class="sourceCode" id="cb14"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb14-1"><a href="#cb14-1" aria-hidden="true" tabindex="-1"></a>output_layer <span class="op">=</span> nn.Linear(in_features<span class="op">=</span><span class="dv">84</span>, out_features<span class="op">=</span><span class="dv">10</span>)</span></code></pre></div>
<h2 data-number="3.5" id="create-lenet-5"><span
class="header-section-number">3.5</span> Create LeNet-5</h2>
<p>LeNet-5 is a simple but famous CNN model. It contains 5
(convolutional or fully-connected) layers. Here, we choose it as our CNN
model. Its architecture is shown in figure 6.</p>
<center>
<img src="https://www.researchgate.net/profile/Vladimir_Golovko3/publication/313808170/figure/fig3/AS:552880910618630@1508828489678/Architecture-of-LeNet-5.png">
<p>
Figure 6. Architecture of LeNet-5
</p>
</center>
<p>The layers of LeNet-5 are summarized here:</p>
<ol start="0" type="1">
<li>Input image: 3x32x32</li>
<li>Conv layer:
<ul>
<li>kernel_size: 5x5</li>
<li>in_channels: 3</li>
<li>out_channels: 6</li>
<li>activation: ReLU</li>
</ul></li>
<li>Max pooling:
<ul>
<li>kernel_size: 2x2</li>
</ul></li>
<li>Conv layer:
<ul>
<li>kernel_size: 5x5</li>
<li>in_channels: 6</li>
<li>out_channels: 16</li>
<li>activation: ReLU</li>
</ul></li>
<li>Max pooling:
<ul>
<li>kernel_size: 2x2</li>
</ul></li>
<li>FC layer:
<ul>
<li>in_features: 16*5*5</li>
<li>out_features: 120</li>
<li>activation: ReLU</li>
</ul></li>
<li>FC layer:
<ul>
<li>in_features: 120</li>
<li>out_features: 84</li>
<li>activation: ReLU</li>
</ul></li>
<li>FC layer:
<ul>
<li>in_features: 84</li>
<li>out_features: 10 (number of classes)</li>
</ul></li>
</ol>
<p><code>model.py</code> create LeNet-5 by PyTorch. First, we create the
2 convolutional layers:</p>
<div class="sourceCode" id="cb15"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb15-1"><a href="#cb15-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> torch.nn <span class="im">as</span> nn</span>
<span id="cb15-2"><a href="#cb15-2" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb15-3"><a href="#cb15-3" aria-hidden="true" tabindex="-1"></a><span class="co"># convolutional layer 1</span></span>
<span id="cb15-4"><a href="#cb15-4" aria-hidden="true" tabindex="-1"></a>conv_layer1 <span class="op">=</span> nn.Sequential(</span>
<span id="cb15-5"><a href="#cb15-5" aria-hidden="true" tabindex="-1"></a>    nn.Conv2d(in_channels<span class="op">=</span><span class="dv">3</span>, out_channels<span class="op">=</span><span class="dv">6</span>, kernel_size<span class="op">=</span>(<span class="dv">5</span>,<span class="dv">5</span>)),</span>
<span id="cb15-6"><a href="#cb15-6" aria-hidden="true" tabindex="-1"></a>    nn.ReLU()),</span>
<span id="cb15-7"><a href="#cb15-7" aria-hidden="true" tabindex="-1"></a>)</span>
<span id="cb15-8"><a href="#cb15-8" aria-hidden="true" tabindex="-1"></a><span class="co"># convolutional layer 2</span></span>
<span id="cb15-9"><a href="#cb15-9" aria-hidden="true" tabindex="-1"></a>conv_layer2 <span class="op">=</span> nn.Sequential(</span>
<span id="cb15-10"><a href="#cb15-10" aria-hidden="true" tabindex="-1"></a>    nn.Conv2d(in_channels<span class="op">=</span><span class="dv">6</span>, out_channels<span class="op">=</span><span class="dv">16</span>, kernel_size<span class="op">=</span>(<span class="dv">5</span>,<span class="dv">5</span>)),</span>
<span id="cb15-11"><a href="#cb15-11" aria-hidden="true" tabindex="-1"></a>    nn.ReLU()),</span>
<span id="cb15-12"><a href="#cb15-12" aria-hidden="true" tabindex="-1"></a>)</span></code></pre></div>
<p>Then, the 3 fully connected layers:</p>
<div class="sourceCode" id="cb16"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb16-1"><a href="#cb16-1" aria-hidden="true" tabindex="-1"></a><span class="co"># fully connected layer 1</span></span>
<span id="cb16-2"><a href="#cb16-2" aria-hidden="true" tabindex="-1"></a>fc_layer1 <span class="op">=</span> nn.Sequential(</span>
<span id="cb16-3"><a href="#cb16-3" aria-hidden="true" tabindex="-1"></a>    nn.Linear(in_features<span class="op">=</span><span class="dv">16</span><span class="op">*</span><span class="dv">5</span><span class="op">*</span><span class="dv">5</span>, out_features<span class="op">=</span><span class="dv">120</span>),</span>
<span id="cb16-4"><a href="#cb16-4" aria-hidden="true" tabindex="-1"></a>    nn.ReLU(),</span>
<span id="cb16-5"><a href="#cb16-5" aria-hidden="true" tabindex="-1"></a>)</span>
<span id="cb16-6"><a href="#cb16-6" aria-hidden="true" tabindex="-1"></a><span class="co"># fully connected layer 2</span></span>
<span id="cb16-7"><a href="#cb16-7" aria-hidden="true" tabindex="-1"></a>fc_layer2 <span class="op">=</span> nn.Sequential(</span>
<span id="cb16-8"><a href="#cb16-8" aria-hidden="true" tabindex="-1"></a>    nn.Linear(in_features<span class="op">=</span><span class="dv">120</span>, out_features<span class="op">=</span><span class="dv">84</span>),</span>
<span id="cb16-9"><a href="#cb16-9" aria-hidden="true" tabindex="-1"></a>    nn.ReLU(),</span>
<span id="cb16-10"><a href="#cb16-10" aria-hidden="true" tabindex="-1"></a>)</span>
<span id="cb16-11"><a href="#cb16-11" aria-hidden="true" tabindex="-1"></a><span class="co"># fully connected layer 3</span></span>
<span id="cb16-12"><a href="#cb16-12" aria-hidden="true" tabindex="-1"></a>fc_layer3 <span class="op">=</span> nn.Linear(in_features<span class="op">=</span><span class="dv">84</span>, out_features<span class="op">=</span><span class="dv">10</span>)</span></code></pre></div>
<p>Finally, combine them as LeNet-5. Don’t forget flatten layer before
FC layers.</p>
<div class="sourceCode" id="cb17"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb17-1"><a href="#cb17-1" aria-hidden="true" tabindex="-1"></a>LeNet5 <span class="op">=</span> nn.Sequential(</span>
<span id="cb17-2"><a href="#cb17-2" aria-hidden="true" tabindex="-1"></a>    conv_layer1,</span>
<span id="cb17-3"><a href="#cb17-3" aria-hidden="true" tabindex="-1"></a>    nn.MaxPool2d(kernel_size<span class="op">=</span>(<span class="dv">2</span>,<span class="dv">2</span>)),</span>
<span id="cb17-4"><a href="#cb17-4" aria-hidden="true" tabindex="-1"></a>    conv_layer2,</span>
<span id="cb17-5"><a href="#cb17-5" aria-hidden="true" tabindex="-1"></a>    nn.MaxPool2d(kernel_size<span class="op">=</span>(<span class="dv">2</span>,<span class="dv">2</span>)),</span>
<span id="cb17-6"><a href="#cb17-6" aria-hidden="true" tabindex="-1"></a>    nn.Flatten(), <span class="co"># flatten</span></span>
<span id="cb17-7"><a href="#cb17-7" aria-hidden="true" tabindex="-1"></a>    fc_layer1,</span>
<span id="cb17-8"><a href="#cb17-8" aria-hidden="true" tabindex="-1"></a>    fc_layer2,</span>
<span id="cb17-9"><a href="#cb17-9" aria-hidden="true" tabindex="-1"></a>    fc_layer3</span>
<span id="cb17-10"><a href="#cb17-10" aria-hidden="true" tabindex="-1"></a>)</span></code></pre></div>
<h1 data-number="4" id="train-the-model"><span
class="header-section-number">4</span> Train the Model</h1>
<p>After create the network, we will teach it how to distinguish images
between different classes. Intuitively, the teaching is achieved by
showing it the images in train set, and telling it which classes they
belong to. The network will gradually learn the concepts, such as
‘bird’, ‘cat’, ‘dog’, etc., just like how human children learn. This
part of code is in <code>train.py</code>.</p>
<p>First, we import our model <code>LeNet5</code>, and define the loss
function and optimization method. Here, we use cross-entropy loss, which
is designed for classification tasks. This loss measure how similar your
prediction is to the correct answer (ground truth). The closer your
prediction is to the correct one, the smaller this loss is. To minimize
this loss, we need an optimizer. Here, we use stochastic gradient
descent (SGD) method as optimizer.</p>
<div class="sourceCode" id="cb18"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb18-1"><a href="#cb18-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> model <span class="im">import</span> LeNet5</span>
<span id="cb18-2"><a href="#cb18-2" aria-hidden="true" tabindex="-1"></a>model <span class="op">=</span> LeNet5</span>
<span id="cb18-3"><a href="#cb18-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb18-4"><a href="#cb18-4" aria-hidden="true" tabindex="-1"></a>loss_fn <span class="op">=</span> nn.CrossEntropyLoss()</span>
<span id="cb18-5"><a href="#cb18-5" aria-hidden="true" tabindex="-1"></a>optimizer <span class="op">=</span> optim.SGD(model.parameters(), lr<span class="op">=</span><span class="fl">0.001</span>, momentum<span class="op">=</span><span class="fl">0.9</span>)</span></code></pre></div>
<p>When training a network, the most important parameter is learning
rate. In above example, learning rate <code>lr</code> is 0.001. To train
model successfully, you need a proper learning rate. If learning rate is
too small, your loss will converge very slowly. If learning rate is too
big, loss may not converge at all.</p>
<p>Then we start training. The training usually takes minutes to hours.
Once you finish looping over the dataset one time, you finished one
epoch. A successful train usually has multiple epochs. Following example
trains the network for 2 epochs.</p>
<div class="sourceCode" id="cb19"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb19-1"><a href="#cb19-1" aria-hidden="true" tabindex="-1"></a><span class="co"># training</span></span>
<span id="cb19-2"><a href="#cb19-2" aria-hidden="true" tabindex="-1"></a>num_epoch <span class="op">=</span> <span class="dv">2</span></span>
<span id="cb19-3"><a href="#cb19-3" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> epoch <span class="kw">in</span> <span class="bu">range</span>(num_epoch):  </span>
<span id="cb19-4"><a href="#cb19-4" aria-hidden="true" tabindex="-1"></a>    running_loss <span class="op">=</span> <span class="fl">0.0</span></span>
<span id="cb19-5"><a href="#cb19-5" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> i, batch <span class="kw">in</span> <span class="bu">enumerate</span>(trainloader, <span class="dv">0</span>):</span>
<span id="cb19-6"><a href="#cb19-6" aria-hidden="true" tabindex="-1"></a>        <span class="co"># get the images; batch is a list of [images, labels]</span></span>
<span id="cb19-7"><a href="#cb19-7" aria-hidden="true" tabindex="-1"></a>        images, labels <span class="op">=</span> batch</span>
<span id="cb19-8"><a href="#cb19-8" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-9"><a href="#cb19-9" aria-hidden="true" tabindex="-1"></a>        optimizer.zero_grad() <span class="co"># zero the parameter gradients</span></span>
<span id="cb19-10"><a href="#cb19-10" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-11"><a href="#cb19-11" aria-hidden="true" tabindex="-1"></a>        <span class="co"># get prediction</span></span>
<span id="cb19-12"><a href="#cb19-12" aria-hidden="true" tabindex="-1"></a>        outputs <span class="op">=</span> model(images)</span>
<span id="cb19-13"><a href="#cb19-13" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-14"><a href="#cb19-14" aria-hidden="true" tabindex="-1"></a>        <span class="co"># compute loss</span></span>
<span id="cb19-15"><a href="#cb19-15" aria-hidden="true" tabindex="-1"></a>        loss <span class="op">=</span> loss_fn(outputs, labels)</span>
<span id="cb19-16"><a href="#cb19-16" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-17"><a href="#cb19-17" aria-hidden="true" tabindex="-1"></a>        <span class="co"># reduce loss</span></span>
<span id="cb19-18"><a href="#cb19-18" aria-hidden="true" tabindex="-1"></a>        loss.backward()</span>
<span id="cb19-19"><a href="#cb19-19" aria-hidden="true" tabindex="-1"></a>        optimizer.step()</span>
<span id="cb19-20"><a href="#cb19-20" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-21"><a href="#cb19-21" aria-hidden="true" tabindex="-1"></a>        <span class="co"># print statistics</span></span>
<span id="cb19-22"><a href="#cb19-22" aria-hidden="true" tabindex="-1"></a>        running_loss <span class="op">+=</span> loss.item()</span>
<span id="cb19-23"><a href="#cb19-23" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> i <span class="op">%</span> <span class="dv">500</span> <span class="op">==</span> <span class="dv">499</span>:  <span class="co"># print every 500 mini-batches</span></span>
<span id="cb19-24"><a href="#cb19-24" aria-hidden="true" tabindex="-1"></a>            <span class="bu">print</span>(<span class="st">&#39;[</span><span class="sc">%d</span><span class="st">, </span><span class="sc">%5d</span><span class="st">] loss: </span><span class="sc">%.3f</span><span class="st">&#39;</span> <span class="op">%</span></span>
<span id="cb19-25"><a href="#cb19-25" aria-hidden="true" tabindex="-1"></a>                  (epoch <span class="op">+</span> <span class="dv">1</span>, i <span class="op">+</span> <span class="dv">1</span>, running_loss <span class="op">/</span> <span class="dv">500</span>))</span>
<span id="cb19-26"><a href="#cb19-26" aria-hidden="true" tabindex="-1"></a>            running_loss <span class="op">=</span> <span class="fl">0.0</span></span>
<span id="cb19-27"><a href="#cb19-27" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-28"><a href="#cb19-28" aria-hidden="true" tabindex="-1"></a><span class="co">#save our model to a file:</span></span>
<span id="cb19-29"><a href="#cb19-29" aria-hidden="true" tabindex="-1"></a>torch.save(LeNet5.state_dict(), <span class="st">&#39;model.pth&#39;</span>)</span>
<span id="cb19-30"><a href="#cb19-30" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb19-31"><a href="#cb19-31" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;Finished Training&#39;</span>)</span></code></pre></div>
<h1 data-number="5" id="test-the-model"><span
class="header-section-number">5</span> Test the Model</h1>
<p>After training, our model can classify images now. At first, we show
it several images in test set to see if it can correctly recognize
them.</p>
<div class="sourceCode" id="cb20"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb20-1"><a href="#cb20-1" aria-hidden="true" tabindex="-1"></a>dataiter <span class="op">=</span> <span class="bu">iter</span>(testloader)</span>
<span id="cb20-2"><a href="#cb20-2" aria-hidden="true" tabindex="-1"></a>images, labels <span class="op">=</span> dataiter.<span class="bu">next</span>()</span>
<span id="cb20-3"><a href="#cb20-3" aria-hidden="true" tabindex="-1"></a>predictions <span class="op">=</span> model(images).argmax(<span class="dv">1</span>)</span>
<span id="cb20-4"><a href="#cb20-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb20-5"><a href="#cb20-5" aria-hidden="true" tabindex="-1"></a><span class="co"># show some prediction result</span></span>
<span id="cb20-6"><a href="#cb20-6" aria-hidden="true" tabindex="-1"></a>classes <span class="op">=</span> trainset.classes</span>
<span id="cb20-7"><a href="#cb20-7" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;GroundTruth: &#39;</span>, <span class="st">&#39; &#39;</span>.join(<span class="st">&#39;</span><span class="sc">%5s</span><span class="st">&#39;</span> <span class="op">%</span> classes[i] <span class="cf">for</span> i <span class="kw">in</span> labels))</span>
<span id="cb20-8"><a href="#cb20-8" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;Prediction: &#39;</span>, <span class="st">&#39; &#39;</span>.join(<span class="st">&#39;</span><span class="sc">%5s</span><span class="st">&#39;</span> <span class="op">%</span> classes[i] <span class="cf">for</span> i <span class="kw">in</span> predictions))</span>
<span id="cb20-9"><a href="#cb20-9" aria-hidden="true" tabindex="-1"></a>imshow(torchvision.utils.make_grid(images.cpu()))</span></code></pre></div>
You will see images and output like this. Due to randomness, your
results may be different.
<center>
<img src="image/samples.png">
</center>
<pre><code>GroundTruth:    cat  ship  ship plane
Prediction:    cat  ship plane plane</code></pre>
<p>Next, let us look at how the model performs on the whole dataset.</p>
<div class="sourceCode" id="cb22"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb22-1"><a href="#cb22-1" aria-hidden="true" tabindex="-1"></a><span class="at">@torch.no_grad</span>()</span>
<span id="cb22-2"><a href="#cb22-2" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> accuracy(model, data_loader):</span>
<span id="cb22-3"><a href="#cb22-3" aria-hidden="true" tabindex="-1"></a>    model.<span class="bu">eval</span>()</span>
<span id="cb22-4"><a href="#cb22-4" aria-hidden="true" tabindex="-1"></a>    correct, total <span class="op">=</span> <span class="dv">0</span>, <span class="dv">0</span></span>
<span id="cb22-5"><a href="#cb22-5" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> batch <span class="kw">in</span> data_loader:</span>
<span id="cb22-6"><a href="#cb22-6" aria-hidden="true" tabindex="-1"></a>        images, labels <span class="op">=</span> batch</span>
<span id="cb22-7"><a href="#cb22-7" aria-hidden="true" tabindex="-1"></a>        outputs <span class="op">=</span> model(images)</span>
<span id="cb22-8"><a href="#cb22-8" aria-hidden="true" tabindex="-1"></a>        _, predicted <span class="op">=</span> torch.<span class="bu">max</span>(outputs.data, <span class="dv">1</span>)</span>
<span id="cb22-9"><a href="#cb22-9" aria-hidden="true" tabindex="-1"></a>        total <span class="op">+=</span> labels.size(<span class="dv">0</span>)</span>
<span id="cb22-10"><a href="#cb22-10" aria-hidden="true" tabindex="-1"></a>        correct <span class="op">+=</span> (predicted <span class="op">==</span> labels).<span class="bu">sum</span>().item()</span>
<span id="cb22-11"><a href="#cb22-11" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> correct <span class="op">/</span> total</span>
<span id="cb22-12"><a href="#cb22-12" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-13"><a href="#cb22-13" aria-hidden="true" tabindex="-1"></a>train_acc <span class="op">=</span> accuracy(model, trainloader) <span class="co"># accuracy on train set</span></span>
<span id="cb22-14"><a href="#cb22-14" aria-hidden="true" tabindex="-1"></a>test_acc <span class="op">=</span> accuracy(model, testloader)  <span class="co"># accuracy on test set</span></span>
<span id="cb22-15"><a href="#cb22-15" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb22-16"><a href="#cb22-16" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;Accuracy on the train set: </span><span class="sc">%f</span><span class="st"> </span><span class="sc">%%</span><span class="st">&#39;</span> <span class="op">%</span> (<span class="dv">100</span> <span class="op">*</span> train_acc))</span>
<span id="cb22-17"><a href="#cb22-17" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(<span class="st">&#39;Accuracy on the test set: </span><span class="sc">%f</span><span class="st"> </span><span class="sc">%%</span><span class="st">&#39;</span> <span class="op">%</span> (<span class="dv">100</span> <span class="op">*</span> test_acc))</span></code></pre></div>
<p>The output looks like</p>
<pre><code>Accuracy on the train set: 62.34 %
Accuracy on the test set: 57.23 %</code></pre>
<p>Since we trained for only 2 epochs, the accuracy is not very
high.</p>
<h1 data-number="6" id="load-predefined-cnns"><span
class="header-section-number">6</span> Load Predefined CNNs</h1>
<p>Besides build your own CNN from scratch, you may use predefined
networks in <a
href="https://pytorch.org/hub/research-models/compact">PyTorch Hub</a>.
Predefined models have two major advantages.</p>
<ul>
<li>These models were searched out and tested by previous researchers,
and usually performs better than the one you build from scratch.</li>
<li>All of them are already pre-trained for specific vision tasks, such
as image classification, object detection, face recognition. Usually,
you only need to fine-tune the model a little bit to fit your dataset.
If you are lucky enough, some models may perfect fit the task you are
working on without further training.</li>
</ul>
<p>Here we take ResNet18 as an example to show how to use predefined
models. You may read its <a
href="https://pytorch.org/hub/pytorch_vision_resnet/">document</a> for
more information.</p>
<p>First, load ResNet18 from PyTorch Hub.</p>
<div class="sourceCode" id="cb24"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb24-1"><a href="#cb24-1" aria-hidden="true" tabindex="-1"></a>model <span class="op">=</span> torch.hub.load(<span class="st">&#39;pytorch/vision:v0.10.0&#39;</span>,</span>
<span id="cb24-2"><a href="#cb24-2" aria-hidden="true" tabindex="-1"></a>                       <span class="st">&#39;resnet18&#39;</span>,</span>
<span id="cb24-3"><a href="#cb24-3" aria-hidden="true" tabindex="-1"></a>                        pretrained<span class="op">=</span><span class="va">True</span>)</span></code></pre></div>
<p>You may print the model to check its architecture.</p>
<pre><code>&gt;&gt;&gt; print(model)
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): ...
  (relu): ...
  (maxpool): ...
  (layer1): ...
  (layer2): ...
  (layer3): ...
  (layer4): ...
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)</code></pre>
<p>The out is quite long, but we only care about its first layer and
output layer. Its first layer <code>conv1</code> has
<code>in_channels=3</code>, which means it was designed for colour
images. If you want to apply it to grey images, the first layer need to
replaced by one with <code>in_channels=1</code>.</p>
<div class="sourceCode" id="cb26"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb26-1"><a href="#cb26-1" aria-hidden="true" tabindex="-1"></a>model.conv1 <span class="op">=</span> nn.Conv2d(<span class="dv">1</span>, <span class="dv">64</span>, kernel_size<span class="op">=</span>(<span class="dv">7</span>, <span class="dv">7</span>), stride<span class="op">=</span>(<span class="dv">2</span>, <span class="dv">2</span>), padding<span class="op">=</span>(<span class="dv">3</span>, <span class="dv">3</span>), bias<span class="op">=</span><span class="va">False</span>)</span></code></pre></div>
<p>Its output layer <code>fc</code> has <code>out_features=1000</code>,
which means it was trained on a Dataset with <code>1000</code> classes.
If you want to apply it to other dataset, such as CIFAR10, the output
layer need to replaced by one with <code>out_features=10</code>, because
CIFAR10 only have 10 classes.</p>
<div class="sourceCode" id="cb27"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb27-1"><a href="#cb27-1" aria-hidden="true" tabindex="-1"></a>model.fc <span class="op">=</span> nn.Linear(in_features<span class="op">=</span><span class="dv">512</span>, out_features<span class="op">=</span><span class="dv">10</span>, bias<span class="op">=</span><span class="va">True</span>)</span></code></pre></div>
<p>According to the <a
href="https://pytorch.org/hub/pytorch_vision_resnet/">document</a> of
ResNet, the model expect input images of shape <code>224x224</code>.
Before we feed images into the model, we need to resize images to
224x224.</p>
<div class="sourceCode" id="cb28"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb28-1"><a href="#cb28-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> torchvision <span class="im">import</span> transforms</span>
<span id="cb28-2"><a href="#cb28-2" aria-hidden="true" tabindex="-1"></a>preprocess <span class="op">=</span> transforms.Resize(<span class="dv">256</span>)</span>
<span id="cb28-3"><a href="#cb28-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-4"><a href="#cb28-4" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i, batch <span class="kw">in</span> <span class="bu">enumerate</span>(trainloader, <span class="dv">0</span>):</span>
<span id="cb28-5"><a href="#cb28-5" aria-hidden="true" tabindex="-1"></a>    images, labels <span class="op">=</span> batch</span>
<span id="cb28-6"><a href="#cb28-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-7"><a href="#cb28-7" aria-hidden="true" tabindex="-1"></a>    <span class="co"># resize to fit the input size of resnet18</span></span>
<span id="cb28-8"><a href="#cb28-8" aria-hidden="true" tabindex="-1"></a>    images <span class="op">=</span> preprocess(images)</span>
<span id="cb28-9"><a href="#cb28-9" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-10"><a href="#cb28-10" aria-hidden="true" tabindex="-1"></a>    <span class="co"># feed into model</span></span>
<span id="cb28-11"><a href="#cb28-11" aria-hidden="true" tabindex="-1"></a>    optimizer.zero_grad()</span>
<span id="cb28-12"><a href="#cb28-12" aria-hidden="true" tabindex="-1"></a>    outputs <span class="op">=</span> model(images)</span>
<span id="cb28-13"><a href="#cb28-13" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb28-14"><a href="#cb28-14" aria-hidden="true" tabindex="-1"></a>    <span class="co"># compute loss, back propagation, etc.</span></span>
<span id="cb28-15"><a href="#cb28-15" aria-hidden="true" tabindex="-1"></a>    ...</span></code></pre></div>
<p><strong>Which model should I use?</strong></p>
<p>First, you need to know the task you are working on. For example, if
you are developing an object detection system, then you should only
consider those model developed for object detection, such as YOLOv5. For
image recognition, you should consider ResNet, AlexNet, etc.</p>
<p>Second, the predefined model usually have several variants of
different size. For example, ResNet has five variants – ResNet18,
ResNet34, ResNet50, ResNet101, ResNet152, which contain 18, 34, 50, 101,
152 layers perspectively. Bigger models have more parameters and
modelling capability, but consumes more memory, more computational
resources and more power. Generally, You should choose big model for
difficult tasks and big datasets, and choose small model for easy tasks
and small datasets.</p>
<h1 data-number="7" id="gpu-acceleration"><span
class="header-section-number">7</span> GPU Acceleration</h1>
<p>GPU acceleration plays an important role in reducing training time of
CNNs. Modern CNNs usually contains tons of trainable parameters and are
extremely computationally hungry. It takes hours to days, or even weeks
to training a perfect CNN model. GPU acceleration technique could speed
up training by 10-100 times, compared to CPU. Figure 7 shows a typical
GPU acceleration performance. The acceleration is even more significant
with large batch-size.</p>
<center>
<img src="https://i2.wp.com/raw.githubusercontent.com/dmlc/web-data/master/nnvm-fusion/perf_lenet.png" style="width: 90%">
<p>
Figure 7. Typical GPU acceleration against CPU.
</p>
</center>
<h2 data-number="7.1" id="install-cuda"><span
class="header-section-number">7.1</span> Install CUDA</h2>
<p>To get GPU acceleration, you need to have a computer with NVIDIA GPU
equipped and have <code>cudatoolkit</code> installed. If your own PC
does not have NVIDIA GPU, but you still want GPU acceleration, you may
use computers in COMP laboratories. All PCs in COMP laboratories are
equipped with NVIDIA GPU. Please use following command to install
<code>cudatoolkit</code>.</p>
<pre><code>$ conda install cudatoolkit=11.3 -c pytorch</code></pre>
<p>Then test your <code>PyTorch</code> to check your installation.</p>
<pre><code>$ python -c &quot;import torch; print(torch.cuda.is_available())&quot;
True</code></pre>
<p>You shall see output <code>True</code>.</p>
<h2 data-number="7.2" id="adapt-your-code-for-gpu"><span
class="header-section-number">7.2</span> Adapt your code for GPU</h2>
<p>You also need to modify your code to get the acceleration.
Specifically, you need to move your model and data to GPU.</p>
<p>Move model to GPU:</p>
<div class="sourceCode" id="cb31"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb31-1"><a href="#cb31-1" aria-hidden="true" tabindex="-1"></a>device <span class="op">=</span> torch.device(<span class="st">&#39;cuda:0&#39;</span>) <span class="co"># get your GPU No. 0</span></span>
<span id="cb31-2"><a href="#cb31-2" aria-hidden="true" tabindex="-1"></a>model <span class="op">=</span> model.to(device)        <span class="co"># move model to GPU</span></span></code></pre></div>
<p>Move data to GPU:</p>
<div class="sourceCode" id="cb32"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb32-1"><a href="#cb32-1" aria-hidden="true" tabindex="-1"></a><span class="co"># get some image from loader</span></span>
<span id="cb32-2"><a href="#cb32-2" aria-hidden="true" tabindex="-1"></a>dataiter <span class="op">=</span> <span class="bu">iter</span>(testloader)</span>
<span id="cb32-3"><a href="#cb32-3" aria-hidden="true" tabindex="-1"></a>images, labels <span class="op">=</span> dataiter.<span class="bu">next</span>()</span>
<span id="cb32-4"><a href="#cb32-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb32-5"><a href="#cb32-5" aria-hidden="true" tabindex="-1"></a><span class="co"># move it to GPU</span></span>
<span id="cb32-6"><a href="#cb32-6" aria-hidden="true" tabindex="-1"></a>images <span class="op">=</span> images.to(device)</span>
<span id="cb32-7"><a href="#cb32-7" aria-hidden="true" tabindex="-1"></a>labels <span class="op">=</span> labels.to(device)</span></code></pre></div>
<p>Get the prediction as usual, but the computation is done by GPU and
thus faster.</p>
<div class="sourceCode" id="cb33"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb33-1"><a href="#cb33-1" aria-hidden="true" tabindex="-1"></a><span class="co"># get prediction as usual</span></span>
<span id="cb33-2"><a href="#cb33-2" aria-hidden="true" tabindex="-1"></a>predictions <span class="op">=</span> model(images).argmax(<span class="dv">1</span>).detach()</span>
<span id="cb33-3"><a href="#cb33-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb33-4"><a href="#cb33-4" aria-hidden="true" tabindex="-1"></a><span class="co"># or perform one-step training, if you are training the model</span></span>
<span id="cb33-5"><a href="#cb33-5" aria-hidden="true" tabindex="-1"></a>optimizer.zero_grad()</span>
<span id="cb33-6"><a href="#cb33-6" aria-hidden="true" tabindex="-1"></a>outputs <span class="op">=</span> model(images)</span>
<span id="cb33-7"><a href="#cb33-7" aria-hidden="true" tabindex="-1"></a>loss <span class="op">=</span> loss_fn(outputs, labels)</span>
<span id="cb33-8"><a href="#cb33-8" aria-hidden="true" tabindex="-1"></a>loss.backward()</span>
<span id="cb33-9"><a href="#cb33-9" aria-hidden="true" tabindex="-1"></a>optimizer.step()</span></code></pre></div>
<p>Finally, if you want to print the result, you may transfer the result
back to CPU:</p>
<div class="sourceCode" id="cb34"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb34-1"><a href="#cb34-1" aria-hidden="true" tabindex="-1"></a><span class="co"># transfer results back to CPU and so we can print it</span></span>
<span id="cb34-2"><a href="#cb34-2" aria-hidden="true" tabindex="-1"></a>predictions <span class="op">=</span> predictions.cpu()</span>
<span id="cb34-3"><a href="#cb34-3" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(predictions)</span></code></pre></div>
<p>However, above code only works on PCs with GPU. To let our code be
able to work, no matter GPU is equipped or not, we usually define
<code>device</code> as:</p>
<div class="sourceCode" id="cb35"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb35-1"><a href="#cb35-1" aria-hidden="true" tabindex="-1"></a><span class="cf">if</span> torch.cuda.is_available():</span>
<span id="cb35-2"><a href="#cb35-2" aria-hidden="true" tabindex="-1"></a>    <span class="co"># If GPU is available, use gpu.</span></span>
<span id="cb35-3"><a href="#cb35-3" aria-hidden="true" tabindex="-1"></a>    device <span class="op">=</span> torch.device(<span class="st">&#39;cuda:0&#39;</span>)</span>
<span id="cb35-4"><a href="#cb35-4" aria-hidden="true" tabindex="-1"></a><span class="cf">else</span>:</span>
<span id="cb35-5"><a href="#cb35-5" aria-hidden="true" tabindex="-1"></a>    <span class="co"># If not, use cpu.</span></span>
<span id="cb35-6"><a href="#cb35-6" aria-hidden="true" tabindex="-1"></a>    device <span class="op">=</span> torch.device(<span class="st">&#39;cpu&#39;</span>)</span></code></pre></div>
<h1 data-number="8" id="assignment"><span
class="header-section-number">8</span> Assignment</h1>
<h2 data-number="8.1" id="handwritten-digit-recognition-18-points"><span
class="header-section-number">8.1</span> Handwritten digit recognition
(18 points)</h2>
<p>Modify <code>train.py</code>, and <code>model.py</code> to train a
CNN to recognize hand-written digits in MNIST datasets.</p>
<center>
<img src="https://www.researchgate.net/profile/Steven_Young11/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.png" style="width:50%">
<p>
Figure 8. Example images from MNIST
</p>
</center>
<p>MNIST contains images of digits 0-9 written by human. The task is to
recognize which digit the image represent. All image are grey scale
(only 1 channel), and contains 28x28 pixels.</p>
<p>Your CNN should contain following layers in order.</p>
<ol start="0" type="1">
<li>Input image: 1x28x28</li>
<li>Conv layer:
<ul>
<li>kernel_size: 5x5</li>
<li>out_channels: 16</li>
<li>activation: ReLU</li>
</ul></li>
<li>Max pooling:
<ul>
<li>kernel_size: 2x2</li>
</ul></li>
<li>Conv layer:
<ul>
<li>kernel_size: 3x3</li>
<li>out_channels: 32</li>
<li>activation: ReLU</li>
</ul></li>
<li>Max pooling:
<ul>
<li>kernel_size: 2x2</li>
</ul></li>
<li>Conv layer:
<ul>
<li>kernel_size: 3x3</li>
<li>out_channels: 32</li>
<li>activation: ReLU</li>
</ul></li>
<li>FC layer:
<ul>
<li>in_features: ?? (to be inferred by you)</li>
<li>out_features: 64</li>
<li>activation: ReLU</li>
</ul></li>
<li>FC layer:
<ul>
<li>out_features: ?? (to be inferred by you)</li>
<li>activation: None</li>
</ul></li>
</ol>
<p>Tasks:</p>
<ol type="1">
<li>Answer following questions first (7 points):
<ul>
<li>Assume we set <code>batch_size=8</code>, what shape do above 7
layers’ inputs and outputs have?</li>
<li>How many trainable parameters each layer contains?</li>
</ul></li>
<li>Train a CNN to recognize hand-written digits in MNIST datasets.
<ul>
<li>Modify <code>model.py</code> to create a CNN with architecture
specified above. (7 points)</li>
<li>Modify <code>train.py</code> and <code>dataset.py</code> to train it
on MNIST dataset. Your model should achieve an accuracy higher than 95%
on test set. Usually, more than 3 epochs are enough to achieve this
accuracy. Save your model to file <code>model.pth</code>. (4
points)</li>
</ul></li>
</ol>
<h2 data-number="8.2" id="bonus-fashion-mnist-3-points"><span
class="header-section-number">8.2</span> Bonus: Fashion-MNIST (3
points)</h2>
<p>MNIST is too easy. Convolutional nets can achieve 99%+ accuracy on
MNIST. Fashion-MNIST, containing ten different classes of clothes, is
more challenging than MNIST. Your task is to load predefined CNN - <a
href="https://pytorch.org/hub/pytorch_vision_resnet/">ResNet18</a> from
PyTorch Hub, and train it to achieve 90+% classification accuracy on
Fashion-MNIST. Please modify <code>fashion_mnist.py</code> to complete
this part.</p>
<p>Tasks:</p>
<ol type="1">
<li><p>You may use <code>load_fashion_mnist()</code> in
<code>dataset.py</code> to load Fashion-MNIST.</p></li>
<li><p>Preprocess data:</p>
<ul>
<li>Apply torchvision.transforms.Resize to resize images in
Fashion-MNIST to fit the input size of ResNet18. The input and output
layer of the network also need modification to fit Fashion-MNIST. (0.5
point)</li>
<li>Try data augmentation functions like
torchvision.transforms.RandomHorizontalFlip and
torchvision.transforms.RandomRotation, discuss the influence of the data
augmentation parameters to the final accuracy. (0.5 point)</li>
</ul></li>
<li><p>Train the model, Save the final model to
<code>fashion_mnist.pth</code> (2 points).</p></li>
</ol>
<center>
<img src="https://github.com/zalandoresearch/fashion-mnist/blob/master/doc/img/fashion-mnist-sprite.png?raw=true" style="width:50%">
<p>
Figure 9. Example images from Fashion-MNIST
</p>
</center>
<h2 data-number="8.3" id="submission-instruction"><span
class="header-section-number">8.3</span> Submission instruction</h2>
<p>Your submission should include:</p>
<ol type="1">
<li>A report, containing
<ul>
<li>your answer to above questions</li>
<li>screenshots of your program output.</li>
</ul></li>
<li>All python source file.</li>
<li>The saved models – <code>model.pth</code> and
<code>fashion_mnist.pth</code>.</li>
<li><strong>Do not</strong> upload the datasets used for training.</li>
</ol>
<p>Please submit before <strong>23:59 on April 30 (Sunday)</strong>. You
may submit as many times as you want, but only your latest submission
will be graded.</p>
</body>
</html>
