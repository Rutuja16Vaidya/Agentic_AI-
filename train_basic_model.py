import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("sarojini_naidu_poetry_dataset.csv").head(5)

# Use these exact five poems as the training text.
# Tone labels are loaded from the first 5 rows of the CSV.
train_poems = [
    """
Bangle sellers are we who bear
Our shining loads to the temple fair...
Who will buy these delicate, bright
Rainbow-tinted circles of light?
Lustrous tokens of radiant lives,
For happy daughters and happy wives.

Some are meet for a maiden's wrist,
Silver and blue as the mountain mist,
Some are flushed like the buds that dream
On the tranquil brow of a woodland stream,
Some are aglow wth the bloom that cleaves
To the limpid glory of new born leaves

Some are like fields of sunlit corn,
Meet for a bride on her bridal morn,
Some, like the flame of her marriage fire,
Or, rich with the hue of her heart's desire,
Tinkling, luminous, tender, and clear,
Like her bridal laughter and bridal tear.

Some are purple and gold flecked grey
For she who has journeyed through life midway,
Whose hands have cherished, whose love has blest,
And cradled fair sons on her faithful breast,
And serves her household in fruitful pride,
And worships the gods at her husband's side.
""",
    """
WEAVERS, weaving at break of day,
Why do you weave a garment so gay? . . .
Blue as the wing of a halcyon wild,
We weave the robes of a new-born child.

Weavers, weaving at fall of night,
Why do you weave a garment so bright? . . .
Like the plumes of a peacock, purple and green,
We weave the marriage-veils of a queen.

Weavers, weaving solemn and still,
What do you weave in the moonlight chill? . . .
White as a feather and white as a cloud,
We weave a dead man's funeral shroud.
""",
    """
Is there aught you need that my hands withhold,
Rich gifts of raiment or grain or gold?
Lo! I have flung to the East and West
Priceless treasures torn from my breast,
And yielded the sons of my stricken womb
To the drum-beats of duty, the sabres of doom.

Gathered like pearls in their alien graves
Silent they sleep by the Persian waves,
Scattered like shells on Egyptian sands,
They lie with pale brows and brave, broken hands,
They are strewn like blossoms mown down by chance
On the blood-brown meadows of Flanders and France.

Can ye measure the grief of the tears I weep
Or compass the woe of the watch I keep?
Or the pride that thrills thro’ my heart’s despair
And the hope that comforts the anguish of prayer?
And the far sad glorious vision I see
Of the torn red banners of Victory?

When the terror and tumult of hate shall cease
And life be refashioned on anvils of peace,
And your love shall offer memorial thanks
To the comrades who fought in your dauntless ranks,
And you honour the deeds of the deathless ones
Remember the blood of thy martyred sons!
""",
    """
GOLDEN sun of victory, born
In my life's unclouded morn,
In my lambent sky of love,
May your growing glory prove
Sacred to your consecration,
To my heart and to my nation.
Sun of victory, may you be
Sun of song and liberty.

Padmaja

Lotus-maiden, you who claim
All the sweetness of your name,
Lakshmi, fortune's queen, defend you,
Lotus-born like you, and send you
Balmy moons of love to bless you,
Gentle joy-winds to caress you.
Lotus-maiden, may you be
Fragrant of all ecstasy.

Ranadheera

Little lord of battle, hail
In your newly-tempered mail!
Learn to conquer, learn to fight
In the foremost flanks of right,
Like Valmiki's heroes bold,
Rubies girt in epic gold.
Lord of battle, may you be,
Lord of love and chivalry.

Lilamani

Limpid jewel of delight
Severed from the tender night
Of your sheltering mother-mine,
Leap and sparkle, dance and shine,
Blithely and securely set
In love's magic coronet.
Living jewel, may you be
Laughter-bound and sorrow-free.
""",
    """
You flaunt your beauty in the rose, your glory in the dawn,

Your sweetness in the nightingale, your whiteness in the swan.

You haunt my waking like a dream, my slumber like a moon,
Pervade me like a musky scent, possess me like a tune.

Yet, when I crave of you, my sweet, one tender moment's grace,
You cry, "I sit behind the veil, I cannot show my face."

Shall any foolish veil divide my longing from my bliss?
Shall any fragile curtain hide your beauty from my kiss?

What war is this of Thee and Me? Give o'er the wanton strife,
You are the heart within my heart, the life within my life.
""",
]

X = pd.Series(train_poems)

y = df["Tone"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))

sample_poem = ["""
You flaunt your beauty in the rose, your glory in the dawn,

Your sweetness in the nightingale, your whiteness in the swan.

You haunt my waking like a dream, my slumber like a moon,

Pervade me like a musky scent, possess me like a tune.

Yet, when I crave of you, my sweet, one tender moment's grace,

You cry, "I sit behind the veil, I cannot show my face."

Shall any foolish veil divide my longing from my bliss?

Shall any fragile curtain hide your beauty from my kiss?

What war is this of Thee and Me? Give o'er the wanton strife,

You are the heart within my heart, the life within my life.
"""]

prediction = model.predict(sample_poem)

print("Predicted Tone:", prediction[0])