



#expected batch
# [batch],[seq_time],[features]
# single step prediction model


class model:
    def __init__(self):
        self.layers = 1
        self.attlayers = 1
        self.relu = 1

    def encode(self,single_input):
        # simple expander-encoder
        linear_encoder = self.relu(self.layers(single_input))
        single_encoded_output = self.relu(self.layers(linear_encoder))
        return single_encoded_output

    def decoder(self,single_encoded_output):
        linear_decode = self.relu(self.layers(single_encoded_output))
        decoded_output = self.layers(linear_decode)
        return decoded_output

    def seq3seq(self,in_seq_batch):
        x = self.encode(in_seq_batch)
        x = self.attlayers(x)
        out = self.decoder(x)
        return out

    def autoencode(self,label):
        x = self.encode(label)
        out = self.decoder(x)
        return out

    def forward(self,in_seq_batch,label):
        predction = self.seq3seq(in_seq_batch)
        autoencode = self.autoencode(label)
        return predction,autoencode

def train_loop(data_loader,model,criterion,optimizer,device):

    for seq,label in enumerate(data_loader):
        model.zero_grad()
        total_loss = None

        #predict and train autoencode
        out,encode = model(seq,label)

        #calculate the losses
        loss1 = criterion(out, label.to(device).float())
        loss2 = criterion(encode, label.to(device).float())

        #add losses
        if total_loss is None:
            total_loss = sum([loss1,loss2])
        else:
            total_loss = sum([sum([loss1,loss2]),total_loss])

        #backprop
        total_loss.backward()
        optimizer.step()



