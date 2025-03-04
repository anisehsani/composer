|:pickup_truck:| ComposerModel
==============================

Your Pytorch model and training step must be re-organized as a
:class:`.ComposerModel` in order to use our :class:`.Trainer`.
This interface helps our trainer access the necessary parts of your model
to easily speed up training.

Minimal Example
---------------

To create a trainable torchvision ResNet-18 classifier with a cross-entropy loss,
define the |forward| and |loss| methods.

Notice how the forward pass is still under user control (no magic here!)
and encapsulated together clearly within the architecture.

The trainer takes care of:

-  ``x.to(device), y.to(device)``
-  ``loss.backward()``
-  ``optimizer.zero_grad()``
-  ``optimizer.step()``

As well as other features such as distributed training, numerics,
, and gradient accumulation.

.. code:: python

   import torchvision
   import torch.nn.functional as F

   from composer.models import ComposerModel

   class ResNet18(ComposerModel):

       def __init__(self):
           super().__init__()
           self.model = torchvision.models.resnet18()

       def forward(self, batch): # batch is the output of the dataloader
           # specify how batches are passed through the model
           inputs, _ = batch
           return self.model(inputs)

       def loss(self, outputs, batch):
           # pass batches and `forward` outputs to the loss
           _, targets = batch
           return F.cross_entropy(outputs, targets)

The composer model and can then be passed to our trainer.

.. code:: python

   from composer import Trainer

   model = ResNet18()
   optimizer = optim.Adam(model.parameters(), lr=0.0001)
   train_dataloader # standard pytorch dataloader

   trainer = Trainer(model=model, optimizers=optimizer,
                     train_dataloader=train_dataloader, max_duration='10ep')
   trainer.fit()

Both the |forward| and |loss| methods are provided the ``batch`` directly
from the dataloader. We leave the unpacking of that batch into inputs and targets
to the user, since it can very depending on the task.

We also provide several common classes for various tasks,
specifically:

-  :class:`.ComposerClassifier` - classification tasks with a cross entropy
   loss and accuracy metric.
-  :class:`.TIMM` - creates classification models from the popular `TIMM`_
   library.
-  :class:`.ComposerTransformer` - base class for 🤗 `Transformers`_ models.

.. note::

    Users from other frameworks such as pytorch lightning, may be used to
    defining a ``training_step`` method, which groups the forward and loss
    together. However, many of our algorithmic methods (such as
    label smoothing or selective backprop) need to intercept and modify the
    loss. For this reason, we split into two separate methods.

By convention, we define our pytorch layers in the ``self.model``
attribute of :class:`.ComposerModel`, which we encourage because it's easier
to extract the underlying model for inference when training is
completed. However, this is not enforced, and users can configure the
layers directly in the class if they prefer.

Metrics
-------

To compute metrics during training, implement the methods:

.. code:: python

   def validate (self, batch) -> outputs, targets:
       ...

   def metrics(self, train=False) -> Metrics:
       ...

where ``Metrics`` should be compatible with the torchmetrics package. We
require that the output of :meth:`.ComposerModel.validate` be consumable by
torchmetrics. Specifically, in our trainer loop, the validation loop
does something like this:

.. code:: python

   metrics = model.metrics(train=False)

   for batch in val_dataloader:
       outputs, targets = model.validate(batch)
       metrics.update(outputs, targets)  # implements the torchmetrics interface

   metrics.compute()

A full example of a validation implementation would be:

.. code:: python

   class ComposerClassifier(ComposerModel):

       def __init__(self):
           super().__init__()
           self.model = torchvision.models.resnet18()
           self.train_accuracy = torchmetrics.Accuracy()
           self.val_accuracy = torchmetrics.Accuracy()

         ...

       def validate(self, batch):

           inputs, targets = batch
           outputs = self.model(inputs)
           return outputs, targets

       def metrics(self, train=False):
           # defines which metrics to use in each phase of training
           return self.train_accuracy if train else self.val_accuracy

.. note::

    No need to set ``model.eval()`` or ``torch.no_grad()`` — we take care
    of that in our trainer. ``torchmetrics`` also handles metrics logging
    when using distributed training.


Logging Results
~~~~~~~~~~~~~~~

The trainer automatically logs the results of the metrics and the loss
using all of the ``loggers`` specified by the user. For example, to log
the results to a ``dict``, use the :class:`.InMemoryLogger`.

.. seealso::

    Our guide to :doc:`Logging<trainer/logging>`.


Multiple Metrics
~~~~~~~~~~~~~~~~

To run multiple metrics, wrap them in a :class:`torchmetrics.MetricCollection`.

.. code:: python

   from torchmetrics.collections import MetricCollection

   def metrics(self, train: bool = False) -> Metrics:
       if train:
           return MetricCollection([self.train_loss, self.train_accuracy])
       return MetricCollection([self.val_loss, self.val_accuracy])

.. note::

    We all the metrics provided to the validation dataset. If
    you have multiple eval datasets and different metrics, we recommend
    using :class:`.Evaluator` (see :doc:`Evaluation<trainer/evaluation>`)

Integrations
------------



TIMM
~~~~

Integrate with your favorite `TIMM`_ models with our :class:`.TIMM` class.

.. code:: python

   from composer.models import Timm

   timm_model = Timm(model_name='resnet50', pretrained=True)

BERT Example with 🤗 Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we create a BERT model, loading from 🤗 Transformers,
and create a model that is compatible with our trainer.

.. code:: python

   from transformers import AutoModelForSequenceClassification
   from torchmetrics import Accuracy
   from torchmetrics.collections import MetricCollection

   from composer import ComposerModel
   from composer.models.nlp_metrics import LanguageCrossEntropyLoss

   class ComposerBERT(ComposerModel):
       def __init__(self, num_labels):
           super().__init__()
           # huggingface model
           self.model = AutoModelForSequenceClassification.from_pretrained(
                            'bert-base-uncased',
                            num_labels=num_labels
                        )

           # Metrics
           self.train_loss = LanguageCrossEntropyLoss()
           self.val_loss = LanguageCrossEntropyLoss()
           self.train_acc = Accuracy()
           self.val_acc = Accuracy()

       def forward(self, batch):
           outputs = self.model(**batch)
           return outputs

       def loss(self, outputs, batch):
           return outputs['loss']  # huggingface models output a dictionary

       def validate(self, batch):
           labels = batch.pop('labels')
           output = self.forward(batch)
           output = output['logits']
           return output, labels

       def metrics(self, train: bool = False):
           if train:
               return MetricCollection([self.train_loss, self.train_acc])
           return MetricCollection([self.val_loss, self.val_acc])


.. |forward| replace:: :meth:`~.ComposerModel.forward`
.. |loss| replace:: :meth:`~.ComposerModel.loss`
.. _Transformers: https://huggingface.co/docs/transformers/index
.. _TIMM: https://fastai.github.io/timmdocs/
.. _torchvision: https://pytorch.org/vision/stable/models.html
