from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

ea = EventAccumulator('./output/events.out.tfevents.1757399063.dsw-23997-574b4977-fn5f6.6981.0')
ea.Reload()

# 取出某条标量曲线，例如 softmax_loss
steps, vals = zip(*[(s.step, s.value) for s in ea.Scalars('softmax_loss')])

plt.plot(steps, vals)
plt.xlabel('step')
plt.ylabel('softmax_loss')
plt.show()