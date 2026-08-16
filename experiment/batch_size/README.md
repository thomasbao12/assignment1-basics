I tried to keep batch_size * step_size = 64k

I ended the experiment with batch_size 1 early, since the throughput was so terrible.
it was only doing 50 steps or about 12.8k tokens per second.  
validation loss after 12.8k steps was around 3.

batch_size, step, clock_time, validation_loss
1, 12800, 263s, 3
8, 1600, 33s, 2.5
16, 800, 26s, 2.5
32, 400, 18s, 2.45
64, 200, 18s, 2.6
128, 100, 17s, 2.7 # this is from learning_rate/experiment1
512, 30, 20s, 6

After 1k iterations, or 1.3M tokens
512, 1000, 600s, 1.6 

# Takeaways
Throughput rises steeply from batch 1 to ~32, then plateaus; meanwhile, convergence per token is best around batch 32 with the current hyperparameters, and very large batches require LR/schedule adjustment because they make far fewer optimizer updates.
