This is part of 7.2.3

I ended up getting a validation loss < 1.45 using a learning rate sweep that where
learning rates never diverged.  

With 1e-2, the validation and training loss are unstable and bounce around 2.9
The lack of divergence might be due to gradient clipping

