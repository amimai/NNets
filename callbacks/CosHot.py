import math
from keras import backend as  K
from keras.callbacks import Callback

class CosHotRestart(Callback):

    def __init__(self, nb_epochs=100, nb_cycles=8, Pgain=1.2, verbose=0, hammer=3, LRgain=1, holdover = 0.5, valweight=False, save_model=None):
        if nb_cycles > nb_epochs:
            raise ValueError('nb_epochs has to be lower than nb_cycles.')

        super(CosHotRestart, self).__init__()
        self.verbose = verbose # print messages?
        self.nb_epochs = nb_epochs # length of training
        self.nb_cycles = nb_cycles # number of cycles to perform
        self.period = self.nb_epochs // self.nb_cycles # how often to reset LR curve
        self.nb_digits = len(str(self.nb_cycles))

        self.gain = Pgain # increase of period
        self.prev_epoch = 0
        self.delta_p = 0

        self.LRhist = []
        self.lrgain = LRgain # gain of LR between runs, allows LR annealing

        self.minloss = float('inf')
        self.stored_weights = 0
        self.last_save = 0
        self.hammer = hammer # timperiods between savepoint restarts
        self.hammercount = 0
        self.holdover = holdover # increases length of cycle if current learning rate is increasing model performance
        self.valweight = valweight # use weghted val loss to determine model validity, works around data that tends to poor generalisation
        self.save_model=save_model

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if self.valweight:
            loss = (loss*2+logs.get('val_loss'))/3
        if self.minloss > loss: # save best model for restarts
            print('newloss ', loss)
            self.stored_weights = self.model.get_weights()
            if self.save_model is not None:
                filename = 'ep-{0}_{1}-{2}'.format(epoch,str(logs.get('loss'))[:7],str(logs.get('val_loss'))[:7])
                self.model.save_weights('callbacks/CosHot_weights/{0}'.format(filename))
            self.minloss = loss
            self.period += self.holdover # slow down decay if its working
            self.delta_p +=self.holdover
            self.LRhist.append(K.get_value(self.model.optimizer.lr))
        else:
            self.last_save += 1
        # stop  here if criteria not met
        if epoch == 0 or (epoch + 1 - self.prev_epoch) % int(self.period) != 0: return

        cycle = int(epoch / self.period)
        cycle_str = str(cycle).rjust(self.nb_digits, '0')
        print('cycle = %s' % cycle_str)

        # if its not learning dont restart with best weight, hit it with a stick instead
        if self.period+1 > self.last_save and self.hammercount==0:
            self.hammercount = self.hammer+1
        self.last_save = 0

        # amend period to correct calcilations
        self.period = int((self.period-self.delta_p) * self.gain)
        self.delta_p = 0
        self.prev_epoch = epoch
        print('period = %d' % self.period)

        # Resetting the learning rate
        K.set_value(self.model.optimizer.lr, self.base_lr)


        if self.hammercount != 0:
            self.hammercount -= 1

        # restore best weights achieved in cycle to polish model
        if self.hammercount == 0:
            self.model.set_weights(self.stored_weights)
            print('restoring best model')
        else:
            print('hammering for %d' % self.hammercount)


    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose>3: print ('LR =  ',K.get_value(self.model.optimizer.lr))
        if epoch <= 0: return

        lr = self.schedule(epoch)
        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose > 1:
            print('Epoch %05d: CosHot modifying learning '
                  'rate to %s.' % (epoch + 1, lr))

    def schedule(self, epoch):
        lr = math.pi * (epoch % self.period) / self.period
        lr = self.base_lr / 2 * (math.cos(lr) + 1)
        return lr

    def set_model(self, model):
        self.model = model
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get initial learning rate
        self.base_lr = float(K.get_value(self.model.optimizer.lr))

    def on_train_end(self, logs={}):
        # Set weights to the values from the end of the best cycle
        if len(self.LRhist)>4: # use history of sucsess to generate weighted average
            avg = 0
            for i in range(len(self.LRhist)):
                avg += (i+1)*self.LRhist[i]
            avg = avg/len(self.LRhist)
            print ('average LR sucsess = ',avg)
            #avg = self.base_lr + (sum(self.LRhist[3:]) / len(self.LRhist))
            K.set_value(self.model.optimizer.lr, avg*10)
        print('setting LR to ',K.get_value(self.model.optimizer.lr))
        self.model.set_weights(self.stored_weights)
        self.minloss = float('inf')
        self.stored_weights = 0
