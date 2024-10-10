import sys
import os
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

# Now import pyMBE
import pyMBE

class dendrimerBuilder:
    def __init__(self, particle_inner, particle_outer, central_chain_residues, branch_residues, branching_number, num_generations):
        """
        Initialize dendrimerBuilder with key parameters.
        """
        self.particle_inner = particle_inner  # Type for inner particles (central chain nodes)
        self.particle_outer = particle_outer  # Type for outer particles (branch nodes)
        self.central_chain_residues = central_chain_residues  # Residues for central chain
        self.branch_residues = branch_residues  # Residues for branches
        self.branching_number = branching_number  # Number of branches at each node
        self.num_generations = num_generations  # Number of generations for the dendrimer

        # Initialize the dendrimer map
        self.dendrimer_map = None  # Placeholder for storing dendrimer map once defined

    def define_dendrimer(self, pmb):
        """
        Define the dendrimer structure using pyMBE functions and write to pmb.df.

        Args:
            pmb: An instance of the pyMBE class.

        Returns:
            A dictionary representing the dendrimer map.
        """
        # Define particles using pyMBE
        pmb.define_particle(name=self.particle_inner)
        pmb.define_particle(name=self.particle_outer)

        # Define residues
        # Central chain residues
        for residue_name in self.central_chain_residues:
            pmb.define_residue(
                name=residue_name,
                central_bead=self.particle_inner,
                side_chains=[]
            )

        # Branch residues
        for residue_name in self.branch_residues:
            pmb.define_residue(
                name=residue_name,
                central_bead=self.particle_outer,
                side_chains=[]
            )

        # Define the central molecule (core of the dendrimer)
        central_molecule_name = 'central_chain_molecule'
        pmb.define_molecule(
            name=central_molecule_name,
            residue_list=self.central_chain_residues
        )

        # Define branch molecules
        branch_molecule_name = 'branch_molecule'
        pmb.define_molecule(
            name=branch_molecule_name,
            residue_list=self.branch_residues
        )

        # Build the dendrimer map
        dendrimer_map = {
            'central_molecule': central_molecule_name,
            'branch_molecule': branch_molecule_name,
            'branching_number': self.branching_number,
            'num_generations': self.num_generations,
            'structure': {}
        }

        # Initialize node ID counter
        node_id_counter = 0

        # Initialize the structure with the central chain
        dendrimer_map['structure'][0] = {
            'nodes': [],
            'branches': []
        }

        # Assign integer IDs to nodes in the central chain
        central_nodes = []
        for residue_name in self.central_chain_residues:
            node_id = node_id_counter
            node_id_counter += 1
            central_nodes.append(node_id)
            dendrimer_map['structure'][0]['nodes'].append(node_id)

        # Build branches for each generation
        current_node_list = central_nodes.copy()
        for gen in range(1, self.num_generations + 1):
            dendrimer_map['structure'][gen] = {
                'nodes': [],
                'branches': []
            }
            next_node_list = []
            for parent_node in current_node_list:
                for _ in range(self.branching_number):
                    # Assign a new integer ID for the branch node
                    node_id = node_id_counter
                    node_id_counter += 1
                    dendrimer_map['structure'][gen]['nodes'].append(node_id)
                    # Store the relationship in branches
                    dendrimer_map['structure'][gen]['branches'].append({
                        'from_node': parent_node,
                        'to_node': node_id,
                        'molecule': branch_molecule_name
                    })
                    next_node_list.append(node_id)
            current_node_list = next_node_list.copy()

        # Store the dendrimer map
        self.dendrimer_map = dendrimer_map
        return dendrimer_map

    def create_dendrimer(self, system, pmb, harmonic_bond, ini_pos=None, center_fixed=False):
        """
        Create the dendrimer structure in the EspressoMD system using the dendrimer map.
        """
        # Check parameters consistency
        if self.dendrimer_map is None:
            raise ValueError("Dendrimer map is not defined. Please run define_dendrimer() first.")

        # Generate type_map from pyMBE
        type_map = pmb.get_particle_type_map()
        bond_type = pmb.get_bond_type(harmonic_bond.name)

        # Set initial position to the center of the box if not provided
        if ini_pos is None:
            ini_pos = np.array(system.box_l) / 2
            print(f"Initial position set to center: {ini_pos}")

        # Initialize node positions and IDs
        node_positions = {}
        node_ids = {}
        current_node_list = []
        next_node_list = []

        # Start with the central chain
        central_chain_nodes = []
        position = ini_pos.copy()
        direction = np.array([1, 0, 0])  # Initial direction along x-axis
        bond_length = pmb.get_bond_length(bond_type)

        for node_index in self.dendrimer_map['structure'][0]['nodes']:
            # Create a particle for the central chain node using pyMBE
            particle_name = self.central_chain_residues[0]  # Assuming uniform residues
            particle_type = type_map[pmb.get_particle_type(particle_name)]
            # Add particle to EspressoMD
            pid = system.part.add(pos=position, type=particle_type)
            # Record particle in pmb.df
            pmb.add_particle_to_df(pid=pid, particle_name=particle_name)
            node_ids[node_index] = pid
            central_chain_nodes.append(pid)
            node_positions[node_index] = position.copy()
            current_node_list.append(node_index)

            # Move to the next position along the direction
            position += direction * bond_length.to("nm").magnitude

        # Record bonds along the central chain in pmb.df
        for i in range(len(central_chain_nodes) - 1):
            pmb.add_bond_in_df(
                particle_id1=central_chain_nodes[i],
                particle_id2=central_chain_nodes[i + 1]
            )

        if center_fixed:
            # Fix the center node in space
            system.part[central_chain_nodes[0]].fix = [True, True, True]

        # Build branches for each generation
        for gen in range(1, self.num_generations + 1):
            next_node_list = []
            for parent_node_index in current_node_list:
                parent_pid = node_ids[parent_node_index]
                parent_pos = node_positions[parent_node_index]

                # Generate random points on the sphere surface around the parent node using pyMBE function
                directions = pmb.generate_random_points_in_a_sphere(
                    center=[0, 0, 0],
                    radius=1.0,
                    n_samples=self.branching_number,
                    on_surface=True
                )

                for branch_num in range(self.branching_number):
                    # Calculate new position for the branch node
                    rand_vec = directions[branch_num]
                    position = parent_pos + rand_vec * bond_length.to("nm").magnitude

                    # Create particle for the branch node using pyMBE
                    particle_name = self.branch_residues[0]  # Assuming uniform residues
                    particle_type = type_map[pmb.get_particle_type(particle_name)]
                    pid = system.part.add(pos=position, type=particle_type)
                    # Record particle in pmb.df
                    pmb.add_particle_to_df(pid=pid, particle_name=particle_name)
                    node_id = max(node_ids.keys()) + 1
                    node_ids[node_id] = pid
                    node_positions[node_id] = position.copy()
                    next_node_list.append(node_id)

                    # Record bond in pmb.df
                    pmb.add_bond_in_df(
                        particle_id1=parent_pid,
                        particle_id2=pid
                    )

            current_node_list = next_node_list.copy()

        # After all particles and bonds are recorded, add bonds to EspressoMD system
        pmb.add_bonds_to_espresso(system)

        # Update the dendrimer map with particle IDs
        self.dendrimer_map['particle_ids'] = node_ids

###Test###

pmb = pyMBE.pymbe_library(seed=42) 
# Initialize pyMBE instance


# Define particle types with pyMBE
pmb.define_particle(name='P0', z=0)
pmb.define_particle(name='P1', z=0)
pmb.define_particle(name='tNH', z=1)
pmb.define_particle(name='pNH', z=1)

# Define residues
pmb.define_residue(name='Res0', central_bead='P0', side_chains=[])
pmb.define_residue(name='Res1', central_bead='P1', side_chains=[])

# Define molecules
pmb.define_molecule(name='central_chain_molecule', residue_list=['Res0', 'Res0'])
pmb.define_molecule(name='branch_molecule', residue_list=['Res1'])

# Create an EspressoMD system
import espressomd
system = espressomd.System(box_l=[20, 20, 20])

# Map particle names to type IDs
type_map = {'P0': 0, 'P1': 1, 'tNH': 2, 'pNH': 3}

# Map charges for functionalization
charge_map = {'tNH': 1.0, 'pNH': 1.0}

# Initialize the dendrimer builder
dendrimer = dendrimerBuilder(
    particle_inner='P0',
    particle_outer='P1',
    central_chain_residues=['Res0', 'Res0'],
    branch_residues=['Res1'],
    branching_number=2,  
    num_generations=2
)

# Define the dendrimer structure using pyMBE
dendrimer.define_dendrimer(pmb)

# Create the dendrimer in the EspressoMD system
dendrimer.create_dendrimer(system, pmb, harmonic_bond)
