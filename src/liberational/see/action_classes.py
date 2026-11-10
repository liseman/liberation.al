"""The action model's classes and window length, apart from the model, so code that only reads its
outputs never imports torch (see/stgcn.py has the model). Which classes are which activity is the owner's list
(memory/activity_kinds.py)."""

CLIP = 100  # frames a window holds: 3.3 s at 30 fps

# NTU RGB+D 120's classes, A1 to A120, in label order.
CLASSES = (
    "drink water", "eat meal or snack", "brushing teeth", "brushing hair", "drop", "pick up", "throw",
    "sitting down", "standing up", "clapping", "reading", "writing", "tear up paper", "wear jacket",
    "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat",
    "take off a hat", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping", "jump up",
    "make a phone call", "playing with phone or tablet", "typing on a keyboard", "pointing with finger",
    "taking a selfie", "check time from watch", "rub two hands together", "nod head or bow", "shake head",
    "wipe face", "salute", "put the palms together", "cross hands in front", "sneeze or cough", "staggering",
    "falling", "touch head", "touch chest", "touch back", "touch neck", "nausea or vomiting", "use a fan",
    "punching or slapping other person", "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other",
    "walking apart from each other", "put on headphone", "take off headphone", "shoot at the basket",
    "bounce ball", "tennis bat swing", "juggling table tennis balls", "hush", "flick hair", "thumb up",
    "thumb down", "make ok sign", "make victory sign", "staple book", "counting money", "cutting nails",
    "cutting paper", "snapping fingers", "open bottle", "sniff", "squat down", "toss a coin", "fold paper",
    "ball up paper", "play magic cube", "apply cream on face", "apply cream on hand back", "put on bag",
    "take off bag", "put something into a bag", "take something out of a bag", "open a box",
    "move heavy objects", "shake fist", "throw up cap or hat", "hands up", "cross arms", "arm circles",
    "arm swings", "running on the spot", "butt kicks", "cross toe touch", "side kick", "yawn", "stretch oneself",
    "blow nose", "hit other person with something", "wield knife towards other person",
    "knock over other person", "grab other person's stuff", "shoot at other person with a gun", "step on foot",
    "high-five", "cheers and drink", "carry something with other person", "take a photo of other person",
    "follow other person", "whisper in other person's ear", "exchange things with other person",
    "support somebody with hand", "finger-guessing game",
)  # fmt: skip
