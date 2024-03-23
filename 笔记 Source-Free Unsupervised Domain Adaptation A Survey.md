###### Abstract

Unsupervised domain adaptation (UDA) via deep learning has attracted appealing attention for tackling domain-shift problems caused by distribution discrepancy across different domains. Existing UDA approaches highly depend on the accessibility of source domain data, which is usually limited in practical scenarios due to privacy protection, data storage and transmission cost, and computation burden. To tackle this issue, many source-free unsupervised domain adaptation (SFUDA) methods have been proposed recently, which perform knowledge transfer from a pre-trained source model to unlabeled target domain with source data inaccessible. A comprehensive review of these works on SFUDA is of great significance. In this paper, we provide a timely and systematic literature review of existing SFUDA approaches from a technical perspective. Specifically, we categorize current SFUDA studies into two groups, i.e., white-box SFUDA and black-box SFUDA, and further divide them into finer subcategories based on different learning strategies they use. We also investigate the challenges of methods in each subcategory, discuss the advantages/disadvantages of white-box and black-box SFUDA methods, conclude the commonly used benchmark datasets, and summarize the popular techniques for improved generalizability of models learned without using source data. We finally discuss several promising future directions in this field.

###### Index Terms:

Domain adaptation, source-free, unsupervised learning, survey.



Intra-domain patch-level self-supervision module (IPSM) Intra-domain patch-level self-supervision module (IPSM) 的目的是为了在无监督域自适应（UDA）中提高伪标签的利用率，尤其是在情景感知分割任务中。该方法解决了在大域间隔条件下分离简单和困难样本的挑战。它通过熵基排序来将目标域的样本分为易分组和难分组，并使用一种对抗机制来减少域间或域内的差距。下面详细解释 IPSM 的工作机制：

1. **生成熵图 (Entropy Maps) 和特征图 (Feature Maps)**:
   - 对于给定的目标域数据，模型首先生成特征图。
   - 每个特征图对应的熵图通过对模型预测输出应用 softmax 函数和后续处理计算得到。熵图中每个像素的熵值表示该像素分类的不确定性，熵值越高表示不确定性越大。

2. **熵排序 (Entropy Ranking)**:
   - 使用熵值将每个样本（图像）的熵图分割成多个小块或者补丁。
   - 这些补丁根据其位置被分类为 K×K 类别。
   - 通过计算每个补丁的平均熵值，可以将补丁分为“简单”和“困难”两组，简单的补丁具有较低的平均熵值，而困难的补丁则相反。

3. **补丁级自我监督 (Patch-level Self-supervision)**:
   - 对于简单和困难的补丁，分别生成预测图 \( I_t^o \) 和 \( I_t^' \)。
   - 然后训练一个判别器 \( D \)，它通过学习区分简单和困难的补丁来减少它们之间的差距。这一步称为对抗性训练 (adversarial learning)。

4. **对抗性损失 (ADV loss)**:
   - 判别器 \( D \) 的目的是最大化简单补丁与困难补丁在判别器输出中的差异。
   - 使用对抗性损失来优化模型，使得模型能够更好地区分这两类补丁，并因此减少它们之间的差距。

5. **公式 (1), (13), (14), (15) 的解释**:
   - 公式 (13) 描述了如何计算目标图像补丁的平均熵值。
   - 公式 (14) 表示熵排序的过程，其中 \( Rank(\cdot) \) 是基于熵值的排序函数。
   - 公式 (15) 定义了对抗性损失，它用于训练判别器 \( D \) 以区分简单和困难的补丁。这个损失鼓励模型将简单补丁的预测偏向于真实标签，而将困难补丁的预测偏向于与简单补丁预测不同的方向。

通过这种方法，可以在不可用源域数据的情况下，利用目标域内的数据自我监督，以减少目标域内部的样本差异，并提高UDA任务中模型的准确性和鲁棒性。

```
Instead of generating source-like images directly, some studies propose to align feature
prototypes or feature distribution of source data [40, 41, 42, 43, 44] with those in the target domain. Specifically, Qiu et al. [40] generate feature prototypes for each source category based on a conditional generator and produce pseudo-labels for the target data. The
cross-domain prototype adaptation is achieved by aligning the features derived from
pseudo-labeled target samples to source prototype with the same category label via contrastive learning. Tian et al. [41] construct a virtual domain by simply sampling from an
approximated gaussian mixture model (GMM) to mimic unseen source domain distribution. In terms of adaptation procedure, they reduce the distribution gap between the constructed virtual domain and the target domain via adversarial training, thus bypassing inaccessible source domain. Their practice is based on the assumption that the feature prototype of each category can be mined from each row of the source classifier’ weights [45].
With the same assumption, Ding et al. [42] leverage such source classifier weights and reliable target pseudo-labels derived by spherical k-means clustering to estimate source feature distribution. After that, proxy source data can be sampled from the estimated source
distribution, and a conventional domain adaptation strategy [46] is used to explicitly perform cross-domain feature distribution alignment. Stan et al. [43, 44] propose to first generate a prototypical distribution representing the source data in an embedding feature
space via GMM, and then perform source-free adaptation by enforcing distribution alignment between source and target domains via sliced Wasserstein distance [47].

什么是近似⾼斯混合模型（GMM） 
⾸先通过GMM⽣成⼀个原型分布， 表
⽰嵌⼊特征空间中的源数据， 然后通过切⽚Wasserstein距离强制源域和⽬ 标域之间的分布对齐来
执⾏⽆源适应[ 47]。

这是啥意思

```

这段内容讨论的是在机器学习领域中，特别是在领域适应（domain adaptation）问题上的一些研究方法。在这个背景下，领域适应是指如何调整或训练一个机器学习模型，使其能够从一个领域（源域）迁移到另一个不同的领域（目标域），即使这两个领域的数据分布可能不同。这种情况在实际应用中非常常见，比如一个模型在一种语言的文本上训练，然后希望将其应用到另一种语言的文本上。

具体到你提到的研究方法：

1. **特征原型（Feature Prototypes）与特征分布的对齐**：一些研究提出，不直接生成与源数据相似的图像，而是通过对齐源数据的特征原型或特征分布与目标域的特征来实现领域适应。特征原型可以视为某一类别数据的特征表示的典型或平均形态。

2. **Qiu et al.**：这项研究通过条件生成器为每个源类别生成特征原型，并为目标数据产生伪标签。通过对比学习，将从伪标签目标样本中得到的特征与相同类别的源原型对齐，从而实现跨域原型适应。

3. **Tian et al.**：通过从一个近似的高斯混合模型（GMM）采样来构建一个虚拟域，模仿未见过的源域分布。他们通过对抗训练减少构建的虚拟域与目标域之间的分布差距，从而绕过无法访问的源域。

4. **GMM（高斯混合模型）**：GMM是一种统计模型，用于表示具有多个子群体的数据集，其中每个子群体的数据分布可以用高斯分布（即正态分布）来近似。在这里，它被用来生成一个原型分布，代表嵌入特征空间中的源数据。

5. **通过切片Wasserstein距离进行的无源适应**：Stan et al. 提出首先通过GMM生成代表源数据的原型分布，然后使用切片Wasserstein距离来强制源域和目标域之间的分布对齐，实现无源适应。切片Wasserstein距离是一种度量两个概率分布之间差异的方法，特别适用于高维数据。通过这种方式，可以在不直接访问源数据的情况下，调整模型以更好地适应目标域。

这些方法都是为了解决当源数据不可访问或希望模型能够泛化到不同但相关的域时的领域适应问题。

2.2.1 Self-Supervised Knowledge Distillation
2.2.1⾃监督知识蒸馏

```
详细解释EMA 
Aug-𝛼 和 Aug- 𝛽 
L𝐾𝐷
```

在这张图片中，描述的是一个用于源无监督域适应的自我监督知识蒸馏方法。下面详细解释图中的每个部分：

1. **EMA (Exponential Moving Average)**:
   EMA通常用于平滑时间序列数据，以此来跟踪变量的趋势。在深度学习模型中，EMA可以用于稳定模型训练，通过对模型参数的一个版本进行平均。在教师网络（Teacher Network）中，参数通常会使用其自身和学生网络（Student Network）参数的EMA来更新。这有助于在迁移学习过程中提高模型的稳定性和性能。

   [什么是EMA？如何使用指数移动平均线和公式 --- What is EMA? How to Use Exponential Moving Average With Formula (investopedia.com)](https://www.investopedia.com/terms/e/ema.asp)

2. **Aug-α 和 Aug-β**:
   这两个术语表示数据增强方法。Aug-α和Aug-β分别代表两种不同的数据增强技术，例如图像的翻转、旋转、平移、噪声添加、扭曲等。数据增强是一种正则化技术，可以通过创造数据的变体来扩充训练集，有助于模型学习更为鲁棒的特征，并防止过拟合。

3. **LKD (Knowledge Distillation Loss)**:
   知识蒸馏损失是一种训练策略，其中一个较小的学生网络被训练以模仿一个较大的教师网络的行为。在这个过程中，学生网络试图学习教师网络的输出或其某些中间表示。LKD是在学生网络的输出和教师网络的输出之间计算的损失函数，目标是使学生网络的预测尽可能接近于教师网络的预测。

图片结构说明：

- 图片展示了一个自我监督学习框架，用于在没有源域数据的情况下进行知识蒸馏。
- 目标数据 \( X_T \) 代表目标域中的数据集。
- 目标数据通过 Aug-α 和 Aug-β 这两种不同的数据增强方法进行增强处理，产生变体。
- 学生网络（Student Network）和教师网络（Teacher Network）都是用来处理这些增强的目标数据。
- 教师网络的参数通过对学生网络的参数进行EMA处理来更新。
- 学生网络通过最小化知识蒸馏损失 LKD 来训练，这个损失函数衡量的是学生网络的输出和经过EMA处理的教师网络输出之间的差异。

整个过程目的是无需源域数据就能让模型适应目标域，这对于在数据访问受限或隐私问题显著的情况下非常有用。通过这种自我监督的知识蒸馏方法，模型能够更好地从目标数据中学习并提高其在目标域的性能。

```
Many studies [50, 51, 52, 23, 53, 54, 55, 56] transfer knowledge learned from source data to
the target model via knowledge distillation in a self-supervised manner, as illustrated in
Fig. 6. In these works, most of them [50, 51, 52, 23, 53] achieve source-free domain adaptation via a mean-teacher scheme for knowledge transfer [57], where the target model not
only learns from unseen target domain but also well preserves source model information.
For instance, Liu et al. [50] propose a self-supervised distillation scheme for automatic
polyp detection. By means of keeping output consistency of weak and strong augmented
polyp images, source knowledge is implicitly transferred to the target model with a mean
teacher strategy [57]. Besides, a diversification flow paradigm is designed to gradually
eliminate the style sensitivity among different domains, further enhancing model robustness towards style diversification. Yang et al. [51] also propose a self-supervised meanteacher approach for knowledge distillation, with a Transformer [58] embedded. This
helps the target model focus on object regions rather than less informative background in
an image, thus improving model generalizability. Assuming that both source and target
images are generated from a domain-invariant space by adding noise perturbations on
each specific domain, Xiong et al. [52] establish a super target domain via augmenting perturbations based on original target domain. The super and original target domains are fed
into a mean-teacher framework, with three consistency regularization terms (w.r.t. image,
instance, and class-wise) introduced for domain alignment. Chen et al. [23] first divide the
target data into clean and noisy subsets guided by a computation loss and regard them as
labeled and unlabeled examples, and then utilize the mean teacher technique to self-generate pseudo-labels for the unlabeled target data for domain adaptation.


你以前的解释太不详细 该有的理解和公式都没说 开始完成：
mean-teacher scheme 是啥 详细解释
diversification flow paradigm是啥 详细解释

```



非常好，我们可以逐一深入探讨这些研究工作及其对域适应问题的不同方法。

1. **Liu et al. [53] 的多教师多学生框架**：
   常规的知识蒸馏方法一般使用一个教师网络指导一个学生网络。而Liu等人提出了一个多教师多学生的框架，这里的**每个教师/学生网络都是用预训练的公共网络初始化的**，这样的网络可能是在单个数据集上预训练的。在这个框架中，构造了一个**图模型来表示样本之间的相似性**，并且教师网络预测的这种相似关系用于通过平均教师技术指导学生网络。这种方法的关键在于，它**利用了多个视角（来自不同教师网络的）**来提供一个更全面和鲁棒的知识基础，同时学生网络从这些教师网络中获得更为丰富的信息。

2. **Yu et al. [54] 的风格和结构规范化以及物理先验约束**：
   Yu等人提出的方法与传统的平均教师范式不同。在他们的工作中，不是通过平均学生网络的权重来蒸馏知识，而是通过**风格和结构规范化以及物理先验约束来从教师网络向学生网络转移知识**。风格规范化可能指的是使**学生网络在不同的风格输入下保持一致性**，而**结构规范化则是确保学生网络能够学习到重要的结构信息**。**物理先验则是基于对世界物理规律的了解**，例如在视觉任务中，物体的几何关系和运动约束可以被用作正则化项来帮助网络学习更加真实和有用的表示。

3. **Tang et al. [55] 的逐步知识蒸馏实现数据无关适应**：
   在Tang等人的研究中，他们采取了一种不同的方法来实现数据无关适应。具体来说，他们首先通过构建的邻域几何结构生成伪标签，然后使用最新轮次得到的伪标签来指导当前训练轮次进行知识转移。这种方法允许模型在没有直接目标数据的情况下，通过从已有的知识和几何结构中推断出目标数据的可能结构来适应新的域。这个过程是渐进的，意味着随着时间的推移，模型通过不断的自我改进和自我评估来调整自己的知识和预测。

每种方法都有其独特的优点和适用场景。例如，多教师多学生框架可以提供多样化的学习信号，可能适用于样本多样性大的任务；而风格和结构规范化可能更适用于需要精细化视觉识别的任务；逐步知识蒸馏则可能适合那些完全没有目标域数据的情况。

这些方法体现了领域适应和知识转移研究中的创新性和多样性，每一种方法都试图以不同的方式来解决如何在不同的数据集或域之间迁移和适应知识的问题。这表明在设计一个鲁棒的学习系统时，考



Y. Grandvalet and Y. Bengio, “Semisupervised learning by entropy
minimization,” Advances in Neural
Information Processing Systems, vol. 17,
2004.





#### 4.2.1 Entropy Minimization Loss

Most SFUDA methods utilize an entropy minimization loss \[[194](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib194)\] to reduce uncertainty of model predictions \[[95](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib95), [195](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib195), [196](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib196), [197](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib197), [198](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib198), [199](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib199), [113](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib113), [112](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib112), [60](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib60), [76](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib76), [28](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib28), [200](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib200), [201](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib201)\]. This simple yet effective strategy encourages the model to generate one-hot predictions for more confident learning.



Entropy minimization is a concept utilized in semi-supervised learning to improve the learning from both labeled and unlabeled data by making the model's predictions as certain (or "confident") as possible. In the context of semi-supervised learning, labeled data is often scarce, so leveraging unlabeled data becomes crucial to enhance the model's performance.

**Entropy in Information Theory**: Entropy is a measure from information theory that quantifies the amount of uncertainty or unpredictability in the predictions of a model. In the context of classification, high entropy reflects a model's uncertainty about which class label to assign to a particular instance. If a model is very certain about its predictions, the entropy would be low. For a classifier, this would ideally be a "one-hot" distribution, where one class has a prediction probability of 1 and the rest have 0.

**Entropy Minimization**: The entropy minimization loss is designed to penalize high entropy predictions and reward low entropy, or more certain, predictions. The intuition is that, if the model is forced to make confident predictions (low entropy), it has to adjust its decision boundaries in a way that clusters the unlabeled data with the labeled instances of each class. This can often align with the true but unknown labels of the unlabeled data, thus improving the model's generalization ability.

**Entropy Minimization in Semi-Supervised Learning**: For semi-supervised learning, the entropy minimization loss can be particularly useful. With limited labeled data, the model may not learn a decision boundary that sufficiently separates the classes. By including the entropy minimization loss, the model can use the unlabeled data to better shape the decision boundary. Unlabeled instances are pushed towards the labeled instances of one class, helping to create more distinct clusters.

**How it Works**: The entropy minimization loss can be incorporated into the loss function of a model. During training, the model not only tries to minimize the prediction error on the labeled data (using standard loss functions like cross-entropy) but also minimizes the entropy of the predictions on the unlabeled data. The overall loss function would be a weighted sum of these two losses.

**Practical Considerations**: In practice, implementing entropy minimization requires careful tuning. If the weight of the entropy loss is too high, the model might make overly confident but incorrect predictions on the unlabeled data, leading to overfitting. Conversely, if the weight is too low, the unlabeled data may not contribute enough to the learning process.

**Conclusion**: Entropy minimization loss is a powerful tool in semi-supervised learning, enabling models to leverage unlabeled data effectively by reducing the uncertainty in the model's predictions. It encourages the model to make more decisive predictions on the unlabeled data, thus providing additional guidance for learning a better decision boundary.

#### 4.2.2 Diversity Enforcing Loss

To prevent predicted labels from collapsing to categories with larger number of samples, many studies leverage a diversity enforcing loss to encourage diverse predictions over target domain \[[95](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib95), [202](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib202), [203](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib203), [81](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib81), [196](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib196), [204](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib204), [205](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib205), [206](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib206)\] . The usual practice is to maximize the entropy of empirical label distribution over the batch-wise average of model predictions.

The concept of a **Diversity Enforcing Loss** is introduced in the field of domain adaptation, particularly within methods that do not have access to source domain data, commonly referred to as Source-Free Unsupervised Domain Adaptation (SFUDA). This type of loss is designed to encourage the model to make diverse predictions across the target domain, especially when the target instances can be divided into sets that are similar or dissimilar to the source.

**Why Diversity Enforcing Loss?**
In a classification task, without enforcing diversity, a model may learn to predict majority classes over minority classes, especially when there is a class imbalance. This phenomenon is known as the model collapsing to the categories with a larger number of samples. Essentially, the model takes the easy way out by mostly predicting the dominant class, neglecting the smaller ones. This leads to poor generalization as the model fails to accurately learn and predict minority class instances.

**How Diversity Enforcing Loss Works:**
Diversity enforcing loss functions work against this by penalizing the model when it makes predictions that lack diversity. This type of loss encourages the model to distribute its predictions more evenly across all classes. In practice, it might involve terms in the loss function that encourage the model's output distribution to match a uniform distribution or the expected class distribution in the target domain.

For example, if a classifier is over-predicting a certain class on the target domain, the diversity loss would penalize this behavior, forcing the model to consider other classes as well. This can be particularly useful when dealing with datasets where some classes are underrepresented.

**Application in SFUDA:**
In the context of SFUDA, diversity enforcing loss helps in the following ways:

1. **Avoiding Biased Predictions**: It ensures that the model does not become biased towards the most frequent labels in the target domain, which is particularly important when source data is not available for reference.

2. **Better Generalization**: By encouraging predictions across a wide range of categories, the model is less likely to overfit to specific features of the target data that are not representative of the underlying task.

3. **Leveraging Unlabeled Data**: It is especially useful in semi-supervised settings where the model needs to learn from a large amount of unlabeled data in the target domain, as it promotes learning about all classes, not just the ones that are easiest to learn.

**Implementation Considerations:**

- **Balancing the Loss**: The diversity enforcing loss usually needs to be balanced with other loss functions (like entropy minimization loss or standard classification loss) to ensure that it does not overpower them, causing the model to ignore the actual data distribution.

- **Hyperparameter Tuning**: The weight of the diversity loss in the overall loss function is a hyperparameter that requires careful tuning, as it can significantly affect the model's performance.

In summary, a diversity enforcing loss is a strategy used in domain adaptation to prevent the model from becoming biased towards the more frequently occurring classes in the target domain. This is accomplished by penalizing predictions that lack diversity, thus encouraging the model to consider a broader range of classes during training.

#### 4.2.3 Label Smoothing Technique

In source-free adaptation studies, a pre-trained source model is generally obtained via training on labeled source data before adaptation stages. Currently, many studies use a label smoothing technique \[[207](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib207), [208](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib208)\] to produce a robust source model \[[95](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib95), [102](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib102), [20](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib20), [31](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib31), [209](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib209), [210](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib210)\]. This technique aims to transform original training labels from hard labels (e.g., 1) to soft labels (e.g., 0.95), which prevents the source model from being over-confident, helping enhance its generalization ability. Also, the experiments have shown that label smoothing can encourage closer representations of training samples from the same category \[[207](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib207)\]. With a more general and robust source model, it is likely to boost adaptation performance on target domain.The label smoothing technique is a regularization method used during the training of a model. It changes the way the model learns by adjusting the targets of the classification task. Let's break down the process:

**Traditional Training with Hard Labels:**
In conventional training of classification models, each training example is usually associated with a hard label. A hard label means that if, for example, you have three classes (cat, dog, and bird), the target output for a picture of a cat would be something like [1, 0, 0], indicating that the model should output 100% probability that the image is a cat and 0% for the other classes.

**Issues with Hard Labels:**
While effective, training on hard labels has its disadvantages. It can lead to overfitting, where the model performs well on the training data but fails to generalize to unseen data. Additionally, it can make the model overly confident in its predictions, sometimes to the detriment of its performance, particularly when the model encounters ambiguous examples or noise.

**Label Smoothing Technique:**
Label smoothing addresses these issues by softening the targets during training. Instead of hard labels, the model is trained on soft labels. Continuing with the above example, a soft label for a cat image might look something like [0.9, 0.05, 0.05], where the sum of the probabilities is still 1, but the model is no longer certain that the image is a cat. This encourages the model to be less confident in its predictions, which can lead to better generalization.

Here's how label smoothing works in more detail:

1. **Adjust the Target Distributions:**
   The true label for a class gets a value slightly less than 1, while the other classes get a small non-zero value, such that the probabilities sum to 1. For example, with three classes and a smoothing parameter \( \epsilon \) of 0.1, the target distribution for a cat image would change from [1, 0, 0] to [0.9, 0.05, 0.05].

2. **Regularization Effect:**
   By preventing the model from predicting the training samples with 100% certainty, label smoothing helps the model to generalize better. It also helps to mitigate issues related to overfitting.

3. **Implementation in Loss Function:**
   The model's loss function is then computed using these soft targets, which has the effect of penalizing overconfident predictions and leading to a more regularized and generalized model.

4. **Generalization to Unseen Data:**
   Label smoothing can help the model perform better on unseen data because it avoids the pitfalls of overconfidence and learns a more distributed representation of the data.

5. **Experiments and Evidence:**
   Experiments have shown that models trained with label smoothing tend to have closer representations of training samples from the same category, which is beneficial for generalization.

In summary, label smoothing is an approach to adjust the confidence of the model, making it more robust and improving its generalization capabilities by modifying the target labels used during training. It's particularly useful in transfer learning and domain adaptation, where the goal is to adapt a model to a new domain it hasn't seen during training.

#### 4.2.4 Model Regularization

Many regularization terms are utilized in existing SFUDA methods by incorporating some prior knowledge. For instance, an early learning regularization \[[40](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib40), [211](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib211), [121](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib121)\] is used to prevent the model from over-fitting to label noise. A stability regularization \[[39](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib39), [212](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib212), [213](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib213), [214](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib214)\] is leveraged to prevent parameters of the target model to deviate from those of the source model. A local smoothness regularization \[[39](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib39), [215](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib215)\] is used to encourage output consistency between the target model and its noise-perturbed counterpart, helping improve robustness of the target model. A mixup regularization \[[110](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib110), [31](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib31), [216](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib216), [115](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib115), [217](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib217)\] is used to enforce prediction consistency between original and augmented data, which can mitigate the negative influence of noisy labels.

Model regularization is a crucial aspect of improving the performance of deep learning models, especially in domain adaptation tasks where the model is trained on a source domain and is expected to perform well on a different target domain. Regularization techniques are used to prevent overfitting and to make the model more generalizable. Let's go through the four types of regularization techniques mentioned:

1. **Early Learning Regularization**:
   - This form of regularization is used to prevent the model from overfitting to noisy labels, which can occur when the model starts to memorize the noise instead of learning the underlying patterns.
   - The idea is to apply more weight to the regularization term at the beginning of the training process, gradually reducing its impact as training progresses.
   - The cited papers propose various approaches to implement this, such as adding a regularization term that discourages the model parameters from deviating too much from their initial values during the early stages of training.

2. **Stability Regularization**:
   - Stability regularization is used to ensure the parameters of the target model do not deviate significantly from those of the source model.
   - The goal is to preserve the knowledge acquired from the source domain while allowing the model to learn new representations from the target domain.
   - Techniques such as L2 regularization on the difference between the source and target model parameters are examples of this approach.

3. **Local Smoothness Regularization**:
   - This regularization encourages output consistency between the target model and its perturbed counterpart. The assumption is that small changes to the input should not drastically change the output.
   - It is usually implemented by penalizing the model if the output for an input and a slightly perturbed version of that input are different, thus enforcing the smoothness of the model's predictions.

4. **Mixup Regularization**:
   - Mixup regularization involves creating new training examples by combining images and labels from the training set in a convex manner.
   - This helps in making the model invariant to interpolations of the training samples and is particularly effective in preventing the model from being too confident and overfitting on noisy labels.
   - The regularization term enforces the prediction consistency between the original and the interpolated or augmented data.

Implementing these regularization techniques can help models become more robust and perform better when applied to new, unseen domains, as they encourage the models not to be overly confident and to generalize better from the training data to the target domain. Each technique introduces additional terms to the loss function that the model aims to minimize, thereby guiding the learning process to favor more generalizable patterns in the data.

#### 4.2.5 Confidence Thresholding

Many studies leverage pseudo-labeling to train the target model in a self-supervised way. Instead of utilizing a manually-designed threshold to identify reliable/confident pseudo-labels, a commonly used strategy is automatically learning the confidence threshold for reliable pseudo-label selection \[[218](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib218)\]. To further tackle the class-imbalance problem, some studies \[[213](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib213), [219](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib219), [76](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib76), [79](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib79), [220](https://ar5iv.labs.arxiv.org/html/2301.00265?_immersive_translate_auto_translate=0#bib.bib220)\] propose to learn dynamic threshold for each category, which provides a fair chance for categories with limited samples to generate pseudo-labels for self-training.



Confidence thresholding in the context of self-supervised learning and domain adaptation is a strategy used to filter and select the most reliable pseudo-labels generated by the model. Pseudo-labeling is a technique where a model uses its own predictions to continue learning in the absence of labels. However, this can lead to noise and errors propagating through the model, especially when the confidence in these pseudo-labels is not managed effectively. Confidence thresholding is utilized to mitigate this issue.

**Confidence Threshold for Reliable Pseudo-label Selection**:
Initially, a confidence threshold is a predefined or dynamically adjusted parameter that determines how confident the model should be in its pseudo-label predictions to consider them for further training. For instance, if the confidence threshold is set to 0.9, only predictions with a confidence score above 90% will be accepted as pseudo-labels.

This threshold can be fixed, where a single value is chosen heuristically or through cross-validation. However, this method does not adapt to the varying difficulty of predicting different classes and can be inefficient, especially in scenarios with class imbalance.

**Learning Dynamic Threshold for Each Category**:
To address the limitations of a fixed threshold, dynamic thresholding learns a unique threshold for each category based on the model's performance and the data distribution. This allows categories with fewer samples or more difficult classification boundaries to have a fair chance of contributing to the model's learning.

The dynamic threshold is often learned by observing the model's prediction scores on the target data and adjusting the threshold per category to maximize some performance metric, such as accuracy or F1-score, on validation data. This approach can be more flexible and better suited for datasets with class imbalance or varying levels of difficulty across classes.

Implementing Confidence Thresholding in Domain Adaptation:
In domain adaptation, especially when the source data is not accessible (Source-Free Unsupervised Domain Adaptation - SFUDA), models are trained using only target data. Here, confidence thresholding is crucial to ensure that the model does not reinforce its mistakes during self-training. By learning confidence thresholds that adapt to each category, the model can improve its predictions iteratively and avoid overfitting to potentially noisy pseudo-labels.

The process typically involves the following steps:

1. **Pseudo-label Generation**: The target model predicts labels for the target domain data.
2. **Confidence Assessment**: For each prediction, the model assesses the confidence of its prediction.
3. **Threshold Application**: Predictions with confidence scores above the learned threshold for their respective category are retained as pseudo-labels.
4. **Model Updating**: The model is updated (trained) using the selected pseudo-labels, improving its predictions iteratively.

The confidence thresholding strategy is critical in SFUDA methods to ensure quality control over the pseudo-labels used for further training the model. It helps in maintaining a balance between exploring the target domain's data distribution and exploiting the learned knowledge without supervision from the source domain. The goal is to enhance the model's generalization ability to new, unseen target data by focusing on high-confidence predictions and reducing the impact of noisy labels.