from collections import defaultdict
import torch, os.path, yaml, re, time
from torch.nn import functional as F
from pathlib import Path
import pandas as pd

class MSA_Dataset:
    def fasta(self, file_path):
        """This method parses a subset of the FASTA format
        https://en.wikipedia.org/wiki/FASTA_format"""
        
        print(f"Parsing fasta '{file_path}'")
        data  = {
            'ur_up_':     [], 'accession':  [],
            'entry_name': [], 'offset':     [],
            'taxonomy':   [], 'sequence':   []
        }

        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                
                if line[0] == '>':
                    key = line[1:]
                    
                    if i == 0:
                        name, offset = key.split("/")
                        ur_up_, acc  = None, None
                    else:
                        ur_up_, acc, name_offset = key.split("|")
                        name, offset             = name_offset.split('/')
                    
                    data['ur_up_'].append(ur_up_)
                    data['accession'].append(acc)
                    data['entry_name'].append(name)
                    data['offset'].append(offset)
                    data['sequence'].append('')
                    data['taxonomy'].append(name.split('_')[1])
                else:
                    data['sequence'][-1] += line
                
                if i and (i % 50000 == 0):
                    print(f"Reached: {i}")

        return pd.DataFrame(data=data)

    def labels(self, labels_file, labels = []):
        """Parses the labels file"""
        print(f"Parsing labels '{labels_file}'")
        with open(labels_file, 'r') as f:
            for i, line in enumerate(f):
                labels.append(line.split(':')[-1].strip())
        return pd.Series(labels)

    def trim(self, full_sequences, focus_columns, sequences = []):
        """Trims the sequences according to the focus columns"""
        for seq in full_sequences:
            seq     = seq.replace('.', '-')
            trimmed = [seq[idx].upper() for idx in focus_columns]
            sequences.append(''.join(trimmed))
        return pd.Series(sequences)

    def encode(self, sequences, alphabet, seq2idx):
        t0 = time.time()
        print(f"Generating {len(sequences)} 1-hot encodings")
        tensors, l = [], len(alphabet)
        for seq in sequences:
            idxseq  = [seq2idx[s] for s in seq]
            tensor  = F.one_hot(torch.tensor(idxseq), l).t().float()
            tensors.append(tensor)
        r = torch.stack(tensors)
        print(f"Generating {len(sequences)} 1-hot encodings. Took {round(time.time() - t0, 3)}s", r.shape)
        return r

    def gen_weights(self, encodings, alphabet, batch_size = 1024):
        print(f"Calculating {len(encodings)} weights...")

        msk_idx = 999 #alphabet.find('-') uncomment if you want '-' to not contribute to weights
        seq_len = encodings.shape[2]
        flat    = encodings.flatten(1)
        batches = flat.shape[0] // batch_size + 1
        weights = []

        for i in range(batches):
            print(f"\tBatch {i}")
            window  =      flat[i * batch_size : (i+1) * batch_size]
            encwin  = encodings[i * batch_size : (i+1) * batch_size]
            smatrix = window @ flat.T                  # Similarity matrix
            seq_len = encwin.argmax(dim=1) != msk_idx  # Mask character `-` do
            seq_len = seq_len.sum(-1).unsqueeze(-1)    #  not contribute to weight
            w_batch = 1.0 / (smatrix / seq_len).gt(0.8).sum(1).float()
            weights.append(w_batch)

        weights = torch.cat(weights) 
        neff    = weights.sum()
        return weights, neff

    def mutants(self, df, mutants_path, col='2500'):
        # name of the column of our interest.
        mdf = pd.read_csv(mutants_path)
        mdf = pd.DataFrame(data={'value': mdf[col].values}, index=mdf['mutant'].values)
        wt_row = df.iloc[0]                # wildtype row in df
        wt_off = wt_row['offset']          # wildtype offset (24-286)
        offset = int(wt_off.split('-')[0]) # left-side offset: 24
        wt_full= wt_row['sequence']
        focus_columns = [idx for idx, char in enumerate(wt_full) if char.isupper()] 

        reg_co  = re.compile("([a-zA-Z]+)([0-9]+)([a-zA-Z]+)")
        mutants = { 'mutation': [], 'sequence': [], 'value': [] }
            
        for i, (k, v) in enumerate(mdf.iterrows()):
            v = v['value']
            _from, _index, _to = reg_co.match(k).groups()
            _index             = int(_index) - offset

            if wt_full[_index].islower():
                continue # we skip the lowercase residues
            
            if wt_full[_index] != _from:
                print("WARNING: Mutation sequence mismatch:", k, "full wt index:", _index)
            
            mutant = wt_full[:_index] + _to + wt_full[_index+1:]
            mutant_trimmed = [mutant[idx] for idx in focus_columns]
            
            mutants['mutation'].append(k)
            mutants['sequence'].append(''.join(mutant_trimmed))
            mutants['value'].append(v)
        return pd.DataFrame(data=mutants)    

class PABP_YEAST(MSA_Dataset):
    def __init__(self, **kwargs):
        self.opts = {**{
            'msa_path':     'data/datasets/pabp_yeast/PABP_YEAST_hmmerbit_plmc_n5_m30_f50_t0.2_r115-210_id100_b48.a2m',
            'mutants_path': 'data/datasets/pabp_yeast/PABP_YEAST_Fields2013-singles.csv',
            'labels_path':  None,
        }, **kwargs}
    
    def prepare(self, **kwargs):
        msa_df = self.fasta(self.opts['msa_path'])
        wildtype_seq  = msa_df.sequence[0] 

        # What wildtype column-positions are we confident about (uppercased chars)
        focus_columns = [idx for idx, char in enumerate(wildtype_seq) if char.isupper()] 

        msa_df['trimmed'] = self.trim(msa_df.sequence, focus_columns)
        
        alphabet = set(''.join(msa_df.trimmed.to_list()))
        print(f"Unique AA's: {len(alphabet)}", ''.join(alphabet))
        seq2idx  = dict(map(reversed, enumerate(alphabet)))

        dataset_tensors = self.encode(msa_df.trimmed, alphabet, seq2idx)

        mutants_df = self.mutants(msa_df, self.opts['mutants_path'], col='log')
        mutants_tensor = self.encode(mutants_df.sequence, alphabet, seq2idx)

        weights_tensor, neff = self.gen_weights(dataset_tensors, str(alphabet))
        msa_df['weight'] = pd.Series(weights_tensor)

        return {
            'msa_df':        msa_df,
            'mut_df':        mutants_df,
            'msa_tensors':   dataset_tensors,
            'msa_weights':   weights_tensor,
            'mut_tensors':   mutants_tensor,
            'wildtype_seq':  wildtype_seq, 
            'focus_columns': focus_columns,
            'alphabet':      alphabet,
            'seq_length':    dataset_tensors.shape[2],
        }
    
class BLAT_ECOLX(MSA_Dataset):
    def __init__(self, **kwargs):
        self.opts = {**{
            'msa_path':     'data/datasets/blat_ecolx/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105.a2m',
            'labels_path':  'data/datasets/blat_ecolx/BLAT_ECOLX_hmmerbit_plmc_n5_m30_f50_t0.2_r24-286_id100_b105_LABELS.a2m',
            'mutants_path': 'data/datasets/blat_ecolx/BLAT_ECOLX_Ranganathan2015.csv'
        }, **kwargs} 
    
    def prepare(self, **kwargs):
        msa_df          = self.fasta(self.opts['msa_path'])
        msa_df['label'] = self.labels(self.opts['labels_path'])

        # First sequence in the dataframe/fasta file is our wildtype.
        wildtype_seq  = msa_df.sequence[0] 

        # What wildtype column-positions are we confident about (uppercased chars)
        focus_columns = [idx for idx, char in enumerate(wildtype_seq) if char.isupper()] 

        msa_df['trimmed'] = self.trim(msa_df.sequence, focus_columns)
        
        alphabet = set(''.join(msa_df.trimmed.to_list()))
        print(f"Unique AA's: {len(alphabet)}", ''.join(alphabet))

        #alphabet = 'ACDEFGHIKLMNPQRSTVWXYZ-'
        seq2idx  = dict(map(reversed, enumerate(alphabet)))

        dataset_tensors = self.encode(msa_df.trimmed, alphabet, seq2idx)

        mutants_df = self.mutants(msa_df, self.opts['mutants_path'])
        mutants_tensor = self.encode(mutants_df.sequence, alphabet, seq2idx)
        
        weights_tensor, neff = self.gen_weights(dataset_tensors, str(alphabet))
        msa_df['weight'] = pd.Series(weights_tensor)

        # --> remember to clear the cache when touching this <--
        return {
            'msa_df':        msa_df,
            'mut_df':        mutants_df,
            'msa_tensors':   dataset_tensors,
            'msa_weights':   weights_tensor,
            'mut_tensors':   mutants_tensor,
            'wildtype_seq':  wildtype_seq, 
            'focus_columns': focus_columns,
            'alphabet':      alphabet,
            'seq_length':    dataset_tensors.shape[2],
        }

class Yielder:
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        key_resolved = self.X#getattr(self, self.key)
        repr = [
            f"Batch yielder obj:",
            f"  i = {self.i}",
            f"  X.shape = {key_resolved.shape} (type: {key_resolved.dtype})",
            # f"  y.shape = {self.y.shape} (type: {self.X.dtype})",
        ]
        return "\n".join(repr)
    

class Dataset:
    def __init__(self, dataset_klass, **kwargs):
        Path("data/cache").mkdir(parents=True, exist_ok=True)
        class_name = dataset_klass.__name__
        kwargs_str = yaml.dump(kwargs)
        kwargs_fnm = re.sub('[^A-Za-z0-9.:]+', '', kwargs_str)
        self.name = f"{class_name}_{kwargs_fnm}".strip('_')

        cache_path = f"data/cache/{self.name}.pkl"

        if not os.path.isfile(cache_path):
            self.data = dataset_klass().prepare(**kwargs)
            torch.save(self.data, cache_path)
            print("Wrote the processed dataset to cache file:", cache_path)
        else:
            print("Cache file found! Loading:", cache_path)
            self.data = torch.load(cache_path)

        for k, v in self.data.items():
            setattr(self, k, v)
    
    def __call__(self, key, batch_size=128):
        """This is a generator"""
        yielder = Yielder(key=key)
        N_train = getattr(self, key).shape[0]
        permut  = torch.randperm(N_train)
        accumulator = defaultdict(list)

        for i, batch_offset in enumerate(range(0, N_train, batch_size)):
            idx = permut[batch_offset:batch_offset + batch_size]
            yielder.i = i
            yielder.X = getattr(self, key)[idx]
            # yielder.y = self.y_train[idx]
            yield yielder, accumulator

if __name__ == "__main__":
    dataset = Dataset(PABP_YEAST)#BLAT_ECOLX)
    # data = BLAT_ECOLX().prepare()
    # blat = BLAT_ECOLX()
    # data = blat.prepare()
