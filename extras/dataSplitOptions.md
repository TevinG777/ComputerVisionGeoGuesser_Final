
Claude seems to think option A is better since it's more generalized (and assumes it will be used on locations it has never seen). Idk how i feel about this. ideally, we would have at least one image from the 15-20 clusters we used. idk, we can try both ig and see which one is better.

# Option A (Conservative Split):
Training set (91 locations):

All 3 images from location 1, 2, 3, 4, 5... → 273 images total
Validation set (20 locations):

All 3 images from location 92, 93, 94... → 60 images total
Test set (19 locations):

All 3 images from location 112, 113, 114... → 57 images total
✅ Result: The model NEVER sees any view of test locations during training

During testing, the model encounters a completely new GPS location it's never seen
This is what will happen in the real game - you'll see a new location

# Option B (Aggressive Split):
Training set:

image1_0.png, image1_1.png from location 1
image2_0.png, image2_2.png from location 2
etc.
Validation set:

image1_2.png from location 1 (different angle!)
image2_1.png from location 2
etc.
⚠️ Result: The model sees OTHER angles from the same GPS location during training

During validation/testing, it's seeing a different view of a place it already knows
This is "easier" and not realistic for your game scenario