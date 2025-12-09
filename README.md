In this repository, there some language models that i created from scratch
For example:
    1. Makemore model
    2. WaveNet
    3. NanoGPT

MakeMore model - It's the simplies language model, it replicates people's names (e.g. Emma, Olivia, Isabella)
The arthitecture:
    6 blocks of Linear layer, Batch normalization layer and Tanh activation function

At the end of the training, we got these loss metrics:
    train 2.080120325088501
    val 2.138594150543213

The generated names of the model:
    carlah.
    amelle.
    khi.
    mrix.
    tatyanna.
    sane.

---

WaveNet model - It's more complex Language model, that also replicates names, but in the WaveNet arthitecture
WaveNet separates the name into chunks, and uses each chunk as memory for the next predicting letter

Metrics:
original (3 character context + 200 hidden neurons, 12K params):
train 2.058
val 2.105

3 layers:
train 1.9781228303909302
val 2.0691566467285156

groupping by 2:
train 1.9538010358810425
val 2.0587151050567627

n_hidden=200:
train 1.8886357545852661
val 2.052396059036255

fixed batch norm layer:
train 1.8652535676956177
val 2.0483429431915283

n_emb = 24, n_hidden = 128
train 1.8772616386413574
val 2.0477468967437744

---

NanoGPT - It's a mini micro nano replication of GPT-3 with 1 million parameters
It replicates shakespeare, and just tries to generate words (in the style of shakespeare)
It was training for 10 minutes on 1 million characters text and on RTX 3060

Architecture:
    Transformer

    384 dimensional embedding layer
    24 self attention layers
    6 FeedForward layers
    Normalization and Dropout layers

    Long story short classical transformer, but without input embedding part


Metrics:
step 0: train loss 4.3568, val loss 4.3565
step 100: train loss 2.6311, val loss 2.6414
step 4900: train loss 1.5579, val loss 1.7386

"Replication of Shakespeare":

sea shallow may not it yours, let his I madne
mast is. 'Then is out
Hatatch tedght for is twoo!

Nurse: Pet is pate:
No; thereat of I thatt thee on: say.

Sirst My hout we mearth othis and sakely,
And death, Poosting hast of all them my sapen.
And biddeen'd marry by shall of comest incane.
The sounter, it is ue.

FRY:

HENRY Who cangre?

LICIO:
Ast potse, all.

Pardon:
Nay, the sin my morere?

Shour part, this mone malk thee did rempty fair,
Is ip my from thas ey, tather father,
The the know the vablicts? and all them wear,
To dill:t if will that many your vanter
I were porracts? O tricl'd me bearsediance do mone's all petruee,
For that his maout: fought sir, friends, puright.
I'll thee, do remouse.

PAMHINDIUS:
If smand there, is for natter too formicre
As tho than batto serve-love! That the eive of olt.

VIRCAHUM:
What fair ham lord: thoub you would shall, Comeo?

Some Woorten from'd: you wam bethan:
What, I'll she siger, soild fladier, joy,
Ound blawd's have to strief my dispan.