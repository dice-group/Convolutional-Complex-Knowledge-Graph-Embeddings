class ConExNeg(torch.nn.Module):
    """
    Variant of ConEx for ablation study.
    """
    def __init__(self, params=None):
        super().__init__()
        self.name = 'ConExNeg'
        self.loss = torch.nn.BCELoss()
        self.param = params
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = params['num_entities']
        self.num_relations = params['num_relations']
        self.kernel_size = params['kernel_size']
        self.num_of_output_channels = params['num_of_output_channels']

        # Embeddings.
        self.emb_ent_real = nn.Embedding(self.param['num_entities'], self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.param['num_entities'], self.embedding_dim)  # imaginary i

        self.emb_rel_real = nn.Embedding(self.param['num_relations'], self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.param['num_relations'], self.embedding_dim)  # imaginary i

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_ent_i = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_rel_real = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_rel_i = torch.nn.Dropout(self.param['input_dropout'])
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(self.embedding_dim)

        # Convolution
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.embedding_dim * 4 * self.num_of_output_channels  # 4 because of 4 real values in 2 complex numbers
        self.fc = torch.nn.Linear(self.fc_num_input, self.embedding_dim * 2)

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(self.embedding_dim * 2)
        self.feature_map_dropout = torch.nn.Dropout2d(self.param['feature_map_dropout'])

    def residual_convolution(self, C_1, C_2):
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # Think of x a n image of two complex numbers.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim)], 2)

        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_head_batch(self, *, e1_idx, rel_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.bn_ent_real(self.emb_ent_real(e1_idx))
        emb_head_i = self.bn_ent_i(self.emb_ent_i(e1_idx))
        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.bn_rel_real(self.emb_rel_real(rel_idx))
        emb_rel_i = self.bn_rel_i(self.emb_rel_i(rel_idx))

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)

        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and all entities.
        real_real_real = torch.mm(a * emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(a * emb_head_real * emb_rel_i, self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(b * emb_head_i * emb_rel_real, self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(b * emb_head_i * emb_rel_i, self.emb_ent_real.weight.transpose(1, 0))
        score = real_real_real + real_imag_imag - imag_real_imag - imag_imag_real

        return torch.sigmoid(score)

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_i.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb, rel_emb

class ConExSum(torch.nn.Module):
    """
    Variant of ConEx for ablation study.
    """
    def __init__(self, params=None):
        super().__init__()
        self.name = 'ConExSum'
        self.loss = torch.nn.BCELoss()
        self.param = params
        self.embedding_dim = self.param['embedding_dim']
        self.num_entities = params['num_entities']
        self.num_relations = params['num_relations']
        self.kernel_size = params['kernel_size']
        self.num_of_output_channels = params['num_of_output_channels']

        # Embeddings.
        self.emb_ent_real = nn.Embedding(self.param['num_entities'], self.embedding_dim)  # real
        self.emb_ent_i = nn.Embedding(self.param['num_entities'], self.embedding_dim)  # imaginary i

        self.emb_rel_real = nn.Embedding(self.param['num_relations'], self.embedding_dim)  # real
        self.emb_rel_i = nn.Embedding(self.param['num_relations'], self.embedding_dim)  # imaginary i

        # Dropouts
        self.input_dp_ent_real = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_ent_i = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_rel_real = torch.nn.Dropout(self.param['input_dropout'])
        self.input_dp_rel_i = torch.nn.Dropout(self.param['input_dropout'])
        # Batch Normalization
        self.bn_ent_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_ent_i = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_real = torch.nn.BatchNorm1d(self.embedding_dim)
        self.bn_rel_i = torch.nn.BatchNorm1d(self.embedding_dim)

        # Convolution
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=self.num_of_output_channels,
                                     kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True)

        self.fc_num_input = self.embedding_dim * 4 * self.num_of_output_channels  # 4 because of 4 real values in 2 complex numbers
        self.fc = torch.nn.Linear(self.fc_num_input, self.embedding_dim * 2)

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        self.bn_conv2 = torch.nn.BatchNorm1d(self.embedding_dim * 2)
        self.feature_map_dropout = torch.nn.Dropout2d(self.param['feature_map_dropout'])

    def residual_convolution(self, C_1, C_2):
        emb_ent_real, emb_ent_imag_i = C_1
        emb_rel_real, emb_rel_imag_i = C_2
        # Think of x a n image of two complex numbers.
        x = torch.cat([emb_ent_real.view(-1, 1, 1, self.embedding_dim),
                       emb_ent_imag_i.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_real.view(-1, 1, 1, self.embedding_dim),
                       emb_rel_imag_i.view(-1, 1, 1, self.embedding_dim)], 2)

        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))
        x = self.feature_map_dropout(x)
        x = x.view(x.shape[0], -1)  # reshape for NN.
        x = F.relu(self.bn_conv2(self.fc(x)))
        return torch.chunk(x, 2, dim=1)

    def forward_head_batch(self, *, e1_idx, rel_idx):
        """
        Given a head entity and a relation (h,r), we compute scores for all entities.
        [score(h,r,x)|x \in Entities] => [0.0,0.1,...,0.8], shape=> (1, |Entities|)
        Given a batch of head entities and relations => shape (size of batch,| Entities|)
        """
        # (1)
        # (1.1) Complex embeddings of head entities and apply batch norm.
        emb_head_real = self.bn_ent_real(self.emb_ent_real(e1_idx))
        emb_head_i = self.bn_ent_i(self.emb_ent_i(e1_idx))
        # (1.2) Complex embeddings of relations and apply batch norm.
        emb_rel_real = self.bn_rel_real(self.emb_rel_real(rel_idx))
        emb_rel_i = self.bn_rel_i(self.emb_rel_i(rel_idx))

        # (2) Apply convolution operation on (1).
        C_3 = self.residual_convolution(C_1=(emb_head_real, emb_head_i),
                                        C_2=(emb_rel_real, emb_rel_i))
        a, b = C_3

        # (3) Apply dropout out on (1).
        emb_head_real = self.input_dp_ent_real(emb_head_real)
        emb_head_i = self.input_dp_ent_i(emb_head_i)
        emb_rel_real = self.input_dp_rel_real(emb_rel_real)
        emb_rel_i = self.input_dp_rel_i(emb_rel_i)

        # (4)
        # (4.1) Hadamard product of (2) and (1).
        # (4.2) Hermitian product of (4.1) and all entities.
        real_real_real = torch.mm(a + emb_head_real * emb_rel_real, self.emb_ent_real.weight.transpose(1, 0))
        real_imag_imag = torch.mm(a + emb_head_real * emb_rel_i, self.emb_ent_i.weight.transpose(1, 0))
        imag_real_imag = torch.mm(b + emb_head_i * emb_rel_real, self.emb_ent_i.weight.transpose(1, 0))
        imag_imag_real = torch.mm(b + emb_head_i * emb_rel_i, self.emb_ent_real.weight.transpose(1, 0))
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

        return torch.sigmoid(score)

    def forward_head_and_loss(self, e1_idx, rel_idx, targets):
        return self.loss(self.forward_head_batch(e1_idx=e1_idx, rel_idx=rel_idx), targets)

    def init(self):
        xavier_normal_(self.emb_ent_real.weight.data)
        xavier_normal_(self.emb_ent_i.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_i.weight.data)

    def get_embeddings(self):
        entity_emb = torch.cat((self.emb_ent_real.weight.data, self.emb_ent_i.weight.data), 1)
        rel_emb = torch.cat((self.emb_rel_real.weight.data, self.emb_rel_i.weight.data), 1)
        return entity_emb, rel_emb