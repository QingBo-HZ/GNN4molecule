How to turn a SMILES string into a molecular graph for Pytorch Geometric


# %% [markdown]
# 1 We define a function that maps an RDKit atom object to a suitable atom feature vector.
# 2 We define a function that maps an RDKit bond object to a suitable bond feature vector.
# 3 We define a function that takes as its input a list of SMILES strings and associated labels and then uses the functions 
# from 1.) and 2.) to create a list of labeled Pytorch Geometric graph objects as its output.

# %%
import torch
print(torch.__version__)                # 查看pytorch安装的版本号
print(torch.cuda.is_available())        # 查看cuda是否可用。True为可用，即是gpu版本pytorch
print(torch.cuda.get_device_name(0))    # 返回GPU型号
print(torch.cuda.device_count())        # 返回可以用的cuda（GPU）数量，0代表一个
print(torch.version.cuda)               # 查看cuda的版本


# %%
# Step 0
# import packages

# general tools
import numpy as np

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

# Pytorch and Pytorch Geometric
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


# %% [markdown]
# 这段代码定义了一个函数   one_hot_encoding  ，用于将输入元素   x   转换为独热编码（one-hot encoding）。独热编码是一种常见的特征处理方法，
# 用于将分类变量转换为数值形式，以便用于机器学习模型。以下是对代码的翻译和详细解释：

# %%
# Step 1
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding

# %%
def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb',\
                                'Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt',\
                                'Hg','Pb','Unknown']
    
    # 隐式氢原子，默认是开的
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # 敲重点
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())] #判断原子芳香与否
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)] #将原子的质量进行归一化处理
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)] #原子的范德华半径进行归一化处理
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)] #共价半径进行归一化处理
    
    # 生成原子的特征向量
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
    # 手性特征编码                                
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    # 隐式氢原子特征编码
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

# %%
# Step 2
def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, \
                                    Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    # 共轭体系：共轭体系是指分子中相邻的原子之间存在交替的单键和双键（或三键），使得电子可以在这些键之间自由移动
    # 共轭体系通常具有特殊的稳定性和化学性质。
    # 共轭键：参与共轭体系的化学键称为共轭键
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    # 判断化学键的描述是否使用立体化学
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)

# %%
# Step 3 Generating labeled Pytorch Geometric Graph Objects
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.
    Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        # 使用 GetAdjacencyMatrix 函数获取分子的邻接矩阵，并找到非零元素的位置（即边的索引）
    
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))

    return data_list

# %%
# Training Loop and Summary

# canonical training loop for a Pytorch Geometric GNN model gnn_model
import torch.nn as nn

# create list of molecular graph objects from list of SMILES x_smiles and list of labels y
data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y)

# create dataloader for training
dataloader = DataLoader(dataset = data_list, batch_size = 2**7)

# define loss function
loss_function = nn.MSELoss()

# define optimiser
optimiser = torch.optim.Adam(gnn_model.parameters(), lr = 1e-3)

# loop over 10 training epochs
for epoch in range(10):

    # set model to training mode
    gnn_model.train()

    # loop over minibatches for training
    for (k, batch) in enumerate(dataloader):

        # compute current value of loss function via forward pass
        output = gnn_model(batch)
        loss_function_value = loss_function(output[:,0], torch.tensor(batch.y, dtype = torch.float32))

        # set past gradient to zero
        optimiser.zero_grad()

        # compute current gradient via backward pass
        loss_function_value.backward()

        # update model weights using gradient and optimisation method
        optimiser.step()


