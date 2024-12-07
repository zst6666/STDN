from .dataloader_taxibj import load_data as load_taxibj
from .dataloader_moving_mnist import load_data as load_mmnist
from .dataloader_sevir import load_data as load_sevir
from .dataloader_NS import load_data as load_NS
from .dataloader_fbs import load_data as load_fds
def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'taxibj':
        return load_taxibj(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'mnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    elif dataname == 'sevir':
        return load_sevir(batch_size, val_batch_size, data_root, num_workers)
    elif dataname =='NS':
        return load_NS(batch_size,val_batch_size,data_root,num_workers)
    elif dataname =='fds':
        return load_fds(batch_size,val_batch_size,data_root,num_workers)