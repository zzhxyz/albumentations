import math
import numpy as np


class Schedule(object):
    def __init__(self, steps=None, a_min=None, a_max=None):
        self.current_step = 0
        self.min = a_min
        self.max = a_max
        self.steps = steps
        self.current_value = np.nan
        self.current_step = 0

    def step(self, current_step=None):
        if self.steps is not None and self.current_step > self.steps:
            raise StopIteration

        if current_step is None:
            self.current_step += 1
        else:
            self.current_step = current_step

    def value(self):
        return np.clip(self.current_value, a_min=self.min, a_max=self.max)

    def reset(self):
        raise NotImplementedError

    def __len__(self):
        if self.steps is None:
            raise ValueError('Steps are none')
        return self.steps

    def __float__(self):
        return self.value()


class ConcatSchedule(Schedule):
    def __init__(self, schedules):
        super(ConcatSchedule, self).__init__()
        if any(sched.step is None for sched in schedules[0:len(schedules) - 1]):
            raise ValueError('Only last schedule can be infinite')

        self.schedule_index = 0
        self.schedules = schedules

    def reset(self):
        self.current_step = 0
        self.schedule_index = 0
        for sched in self.schedules:
            sched.reset()

    def step(self, current_step=None):
        self.current_step += 1
        try:
            self.schedules[self.schedule_index].step(current_step)
        except StopIteration:
            self.schedule_index += 1

    def value(self):
        return self.schedules[self.schedule_index].value()

    def __len__(self):
        return sum([len(sched) for sched in self.schedules])


class LinearSchedule(Schedule):
    def __init__(self, initial_value, increment, steps=None, a_min=None, a_max=None):
        super(LinearSchedule, self).__init__(steps, a_min, a_max)
        self.initial_value = initial_value
        self.increment = increment
        self.current_value = self.initial_value

    def step(self, current_step=None):
        super(LinearSchedule, self).step(current_step)
        self.current_value += self.increment

    def reset(self):
        self.current_value = self.initial_value
        self.current_step = 0


class ExpDecay(Schedule):
    def __init__(self, initial_value, decay, steps=None, a_min=None, a_max=None):
        super(ExpDecay, self).__init__(steps, a_min, a_max)
        self.initial_value = initial_value
        self.decay = decay
        self.steps = steps
        self.min = a_min
        self.max = a_max
        self.current_value = self.initial_value
        self.current_step = 0

    def step(self, current_step=None):
        super(ExpDecay, self).step(current_step)

        new_val = self.initial_value ** math.exp(- self.current_step * self.decay)
        self.current_value = np.clip(new_val, a_min=self.min, a_max=self.max)

    def reset(self):
        self.current_step = 0
        self.current_value = self.initial_value

    def value(self):
        return self.current_value


class CosineRestartsDecay(Schedule):
    def __init__(self, initial_value, period, steps=None, a_min=None, a_max=None):
        super(CosineRestartsDecay, self).__init__(steps, a_min, a_max)
        self.period = period
        self.initial_value = initial_value

    def step(self, current_step=None):
        super(CosineRestartsDecay, self).step(current_step)

        period_fraction = float(self.current_step % self.period) / float(self.period)
        self.current_value = self.initial_value * math.cos(0.5 * math.pi * period_fraction)

    def reset(self):
        self.current_step = 0
        self.current_value = self.initial_value


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def get_values(sched, steps=None):
        values = []
        sched.reset()

        if steps is not None:
            rng = range(steps)
        else:
            rng = range(len(sched))

        for _ in rng:
            values.append(sched.value())
            sched.step()

        return values


    lin = LinearSchedule(0, 2, a_min=0, a_max=50, steps=100)
    cos = CosineRestartsDecay(50, 20, a_min=1, steps=100)
    exp = ExpDecay(50, 0.01, a_min=1, steps=100)

    plt.figure()
    plt.plot(get_values(lin), label='Linear')
    plt.plot(get_values(cos), label='Exp')
    plt.plot(get_values(exp), label='Exp')

    concat = ConcatSchedule([lin, cos, exp])
    plt.figure()
    plt.plot(get_values(concat, 300), label='Concat')

    plt.show()
