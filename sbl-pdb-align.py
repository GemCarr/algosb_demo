#!/usr/bin/env python3
# -W ignore
# adapted from https://gist.github.com/andersx/6354971#file-align-py

import sys
import argparse
import os.path
import pdb
import re
import warnings

#import Bio.PDB
from Bio import PDB
from Bio import SeqIO
#from Bio import pairwise2
from Bio import Align
from Bio import AlignIO
from Bio.Align import PairwiseAligner
from Bio.Align import substitution_matrices
from Bio.Data import IUPACData

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import IUPAC

from SBL import SBL_PDB_toolbox
from SBL import SBL_VMD_toolbox
from SBL.SBL_VMD_toolbox import SBL_VMD_toolbox

from SBL.SBL_PDB_toolbox  import SBL_PDB_toolbox

class Aligner:
    # NB: om obtained from vars(options): keys not set have value None
    def __init__(self, options):
        warnings.simplefilter("ignore")
        self.options = options

        if not ( (len(self.options.fnames) == 1 and len(self.options.chains) > 1) or\
                 (len(self.options.fnames) > 1 and len(self.options.fnames) == len(self.options.chains)) ):
            sys.exit("I need either one pdb file and N chains or N pdb files and N chains (N > 1)")
        if len(self.options.fnames) == 1:
            self.options.fnames = self.options.fnames*len(self.options.chains)
        if len(self.options.fnames) > 2:
            print("WARNING: only 'dump_sequence_chains' method can work with more than 2 chains")

        # default: directory containing the structures
        if self.options.output_dir != None:
            self.odir = self.options.output_dir
        # directory containing the input file0 ... or . by default
        else:
            self.odir = os.path.split(self.options.fnames[0])[0]
            if self.odir == "":
                self.odir = "."
        if not os.path.exists(self.odir):
            cmd = "mkdir %s" % self.odir
            os.system(cmd)

        self.m_atoms_selection = self.options.atoms_selection
        self.m_structures = []
        self.m_models = []
        self.m_chains = []

        self.m_seq_record_iter = []
        self.m_seqs = []

        pdb_parser = PDB.PDBParser(QUIET = True)
        for i in range(0, len(self.options.fnames)):
            f = self.options.fnames[i]
            self.m_structures.append( pdb_parser.get_structure( "structure_1", f) )
            self.m_models.append( self.m_structures[i][0] ) # todo: spec model numder

            self.m_chains.append( self.select_chain_in_model(self.m_models[i], self.options.chains[i]) )
            assert(self.m_chains[i] != None)
            msg = "pdb_id%s and chain %s:" %(os.path.split(f)[1], self.m_chains[i].get_id())
            print(msg)

            # sequences and seq records
            # NB: parse() returns a SeqRecord iterator
            # Try to get the SEQ from the PDB file
            self.m_seq_record_iter.append( SeqIO.parse(f, "pdb-seqres") )
            chain_record = self.select_chain_record(self.m_seq_record_iter[i], self.m_chains[i].get_id())
            if  chain_record is None:
                chain_record = SBL_PDB_toolbox.build_chain_seq_record(self.m_chains[i])
            self.m_seqs.append( chain_record.seq )

            print("Sequence has length:", len(self.m_seqs[i]), "\n")

        if len(self.m_chains) > 2:
            self.dump_sequence_chains()
            exit(1)



    def dump_sequence_chains(self):
        print("+++Unaligned sequences dump:", "unaligned.fasta")
        with open("unaligned.fasta", "w") as f, open("unaligned.txt", "w") as g:
            for i in range(len(self.m_chains)):
                pdbid_i = SBL_PDB_toolbox.get_pdbid(self.options.fnames[i])
                f.write(">" + pdbid_i  +"_"+self.m_chains[i].get_id()+"\n")
                f.write(str(self.m_seqs[i])+"\n")
                g.write(self.options.fnames[i]+" "+self.options.chains[i]+"\n")
                print(pdbid_i, self.m_chains[i].get_id())


    def align_seqs(self):
        # (deprecated)
        aligner = PairwiseAligner(scoring="blastp") # Same set of parameters as BLASTP. Mode: global!
        alignments = aligner.align(self.m_seqs[0], self.m_seqs[1])

        scores = [ali.score for ali in alignments]
        print("+++Seq align with Bio.pairwise2 num alignments with scores:", len(alignments), scores)
        # Alignment(seqA='MES...QAA', seqB='MES...QAA', score=425.0, start=0, end=429) #end:past the end
        best_ali = alignments[ scores.index(max(scores)) ]
        # parwise aligment incompatible with the MSA expected by AlignIO. bravo :(
        #with open("test.fasta", "w") as handle:
        #    count = AlignIO.write(alignments, handle, "fasta")
        SBL_PDB_toolbox.save_pairwise2_alignment(self.odir, best_ali,
                                                 self.options.fnames[0], self.m_chains[0].get_id(),
                                                 self.options.fnames[1], self.m_chains[1].get_id())

    def select_chain_in_model(self, model, chain_id):
        for chain in model:
            if chain.get_id() == chain_id:
                return chain
        return None

    # Exple: SeqRecord(seq=Seq('ADKKLVVATDTAFVPFEFKQGDLYVGFDVDLWAAIAKELKLDYELKPMDFSGII...EPK'), id='1GGG:A', name='1GGG:A', description='UNP:P10344 GLNH_ECOLI', dbxrefs=['UNP:P10344', 'UNP:GLNH_ECOLI'])
    def select_chain_record(self, seq_records, chain_id):
        for record in seq_records:
            cid = re.split(":", record.name)[1]
            if cid == chain_id:
                return record
        return None

    def get_atoms_selection(self, chain, aset):
        atoms = []
        for res in chain:
            if res.get_id()[1] in aset:

                if "CA" in self.m_atoms_selection:
                    atoms.append(res['CA'])
                if "N" in self.m_atoms_selection:
                    atoms.append(res['N'])
                if "C" in self.m_atoms_selection:
                    atoms.append(res['C'])
        return atoms

    def get_resids_in_range(self, chain):
        resids = set()
        for res in chain:
            if res.get_id()[1] in self.atoms_to_be_aligned:
                resids.add(res.get_id()[1])
        return resids

    # perfrom the struct align using the common resids in set_12
    # also rewrite the second structure
    def align_two_structures(self, i_index, j_index):
        model_i = self.m_models[i_index]; model_j = self.m_models[j_index]
        chain_i = self.m_chains[i_index]; chain_j = self.m_chains[j_index]

        resid_i = self.get_resids_in_range(chain_i)
        resid_j = self.get_resids_in_range(chain_j)
        resid_ij = resid_i.intersection(resid_j)
        msg = "\n+++Struct align, num residues chain_1:%d chain_2:%d intersection:%d" % (len(resid_i), len(resid_j), len(resid_ij))
        print(msg)

        # List of atoms to be aligned: CA here
        atoms_i = self.get_atoms_selection(chain_i, resid_ij)
        atoms_j = self.get_atoms_selection(chain_j, resid_ij)

        # Now we initiate the superimposer: NB: rigid motion applied to whole model -- not just chain
        assert(len(atoms_i) == len(atoms_j))
        super_imposer = PDB.Superimposer()
        super_imposer.set_atoms(atoms_i, atoms_j)
        super_imposer.apply(model_j.get_atoms())

        # Print RMSD:
        msg = "lRMSD for the id alignment using %d residues: %.2f" % (len(resid_ij), super_imposer.rms)
        print(msg)

        # Save the aligned version of 1UBQ.pdb
        fname, fname_ext = os.path.splitext(self.options.fnames[j_index])
        ofname = "%s-aligned%s" % (fname, fname_ext)
        io = PDB.PDBIO()
        io.set_structure(self.m_structures[j_index])
        io.save(ofname)
        print("Dumped file " + ofname)
        print("PDB files: ", self.options.fnames[i_index], ofname)
        SBL_VMD_toolbox.save_vmd_state(self.options.fnames[i_index], self.m_chains[i_index].get_id(), \
                                       ofname, self.m_chains[j_index].get_id(), self.odir)


    def run(self):
        self.align_seqs()

        # resid range selection
        if self.options.resid_range is None:
            self.atoms_to_be_aligned = set(range(1, max(len(self.m_seqs[0]), len(self.m_seqs[1]))))
        else:
            assert(len(self.options.resid_range)==2)
            self.atoms_to_be_aligned = set(range(self.options.resid_range[0], self.options.resid_range[1]+1))

        self.align_two_structures(0, 1)


# Main
#i################################################################################
#parser = argparse.ArgumentParser(description='My parser')
parser = argparse.ArgumentParser(description='My parser',formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)

parser.add_argument("-f", "--fnames", nargs='+', dest="fnames", help="Two structures/PDB files")
parser.add_argument("-c","--chains", nargs='+', dest="chains", default=["A", "A"], help="Two chain ids")
parser.add_argument("--range", nargs='+', type=int, dest="resid_range", help="Range of resids to use")
parser.add_argument("--asel", nargs='+', dest="atoms_selection", default=["N", "CA", "C"], help="Names of backbone atoms used in the selection")
parser.add_argument("-a", "--ali", nargs='+', dest="ali", help="Alignment type: s:seq S:struct")
parser.add_argument("--output-dir", dest="output_dir", help="Output directory -- created from options if void")

options = parser.parse_args()
ali = Aligner(options)
ali.run()
