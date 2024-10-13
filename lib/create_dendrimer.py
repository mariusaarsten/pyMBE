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
        Define the dendrimer structure and build the dendrimer map.
        """
        # Build the dendrimer map
        dendrimer_map = {
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
                        'to_node': node_id
                    })
                    next_node_list.append(node_id)
            current_node_list = next_node_list.copy()

        # Store the dendrimer map
        self.dendrimer_map = dendrimer_map
        return dendrimer_map

    def create_dendrimer(self, system, pmb, ini_pos=None, center_fixed=False):
        """
        Create the dendrimer structure in the EspressoMD system using the dendrimer map.
        """
        # Check parameters consistency
        if self.dendrimer_map is None:
            raise ValueError("Dendrimer map is not defined. Please run define_dendrimer() first.")

        # Map particle names to type IDs
        type_map = {'P0': 0, 'P1': 1, 'tNH': 2, 'pNH': 3}  # Adjust as needed

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

        for node_index in self.dendrimer_map['structure'][0]['nodes']:
            # Get the central bead particle name from the residue
            residue_name = self.central_chain_residues[0]  # 'Res0'

            # Retrieve the central bead directly from pmb.df
            residue_df = pmb.df[
                (pmb.df[('pmb_type', '')] == 'residue') & 
                (pmb.df[('name', '')] == residue_name)
            ]
            if not residue_df.empty:
                particle_name = residue_df.iloc[0][('central_bead', '')]
            else:
                raise ValueError(f"Residue '{residue_name}' not found in pyMBE definitions.")

            particle_type = type_map[particle_name]
            # Add particle to EspressoMD
            pid = system.part.add(pos=position, type=particle_type)
            node_ids[node_index] = pid
            central_chain_nodes.append(pid)
            node_positions[node_index] = position.copy()
            current_node_list.append(node_index)

            # Move to the next position along the direction
            # Use the bond length from pyMBE
            if len(central_chain_nodes) > 1:
                # Find bond between previous and current particle types
                prev_particle_name = particle_name  # Since all are the same in this case
                bond_object = pmb.search_bond(prev_particle_name, particle_name)
                bond_length = pmb.get_bond_length(prev_particle_name, particle_name)
            else:
                bond_length = pmb.units.Quantity(1.0, "nm")  # Default bond length for the first particle

            position += direction * bond_length.to("nm").magnitude

        # Apply bonds along the central chain using pyMBE bond definitions
        for i in range(len(central_chain_nodes) - 1):
            pid1 = central_chain_nodes[i]
            pid2 = central_chain_nodes[i + 1]
            particle_name1 = particle_name
            particle_name2 = particle_name
            bond_object = pmb.search_bond(particle_name1, particle_name2)
            if bond_object is not None:
                bond_type_id = bond_object.type
            else:
                raise ValueError(f"Bond not found between {particle_name1} and {particle_name2}")
            system.part.add_bond((bond_type, pid1, pid2))

        if center_fixed:
            # Fix the center node in space
            system.part[central_chain_nodes[0]].fix = [True, True, True]

        # Build branches for each generation
        for gen in range(1, self.num_generations + 1):
            next_node_list = []
            for parent_node_index in current_node_list:
                parent_pid = node_ids[parent_node_index]
                parent_pos = node_positions[parent_node_index]

                # Get the parent particle name
                parent_residue_name = self.central_chain_residues[0] if gen == 1 else self.branch_residues[0]
                parent_residue_df = pmb.df[
                    (pmb.df[('pmb_type', '')] == 'residue') & 
                    (pmb.df[('name', '')] == parent_residue_name)
                ]
                if not parent_residue_df.empty:
                    parent_particle_name = parent_residue_df.iloc[0][('central_bead', '')]
                else:
                    raise ValueError(f"Residue '{parent_residue_name}' not found in pyMBE definitions.")

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

                    # Get the branch particle name
                    residue_name = self.branch_residues[0]  # 'Res1'
                    residue_df = pmb.df[
                        (pmb.df[('pmb_type', '')] == 'residue') & 
                        (pmb.df[('name', '')] == residue_name)
                    ]
                    if not residue_df.empty:
                        particle_name = residue_df.iloc[0][('central_bead', '')]
                    else:
                        raise ValueError(f"Residue '{residue_name}' not found in pyMBE definitions.")

                    # Find bond between parent and branch particle types
                    # Get the bond type from pyMBE
                    bond_object = pmb.search_bond(parent_particle_name, particle_name)
                    if bond_object is not None:
                        bond_type_id = bond_object.type
                    bond_length = pmb.get_bond_length(bond_type_id)
                    position = parent_pos + rand_vec * bond_length.to("nm").magnitude

                    particle_type = type_map[particle_name]
                    pid = system.part.add(pos=position, type=particle_type)
                    node_id = max(node_ids.keys()) + 1
                    node_ids[node_id] = pid
                    node_positions[node_id] = position.copy()
                    next_node_list.append(node_id)

                    # Get the bond type from pyMBE
                    bond_type = pmb.get_bond_type(bond_type_id)
                    # Add bond to the EspressoMD system
                    system.part.add_bond((bond_type, parent_pid, pid))

            current_node_list = next_node_list.copy()

        # Update the dendrimer map with particle IDs
        self.dendrimer_map['particle_ids'] = node_ids




###Test###

pmb = pyMBE.pymbe_library(seed=42) 
# Initialize pyMBE instance
import espressomd
# **1. Define Particles with LJ Parameters**
sigma_value = pmb.units.Quantity(1.0, 'nm')    
epsilon_value = pmb.units.Quantity(1.0, 'kJ') 
cutoff_value = 2.5 * sigma_value

pmb.define_particle(
    name='P0',
    z=0,
    sigma=sigma_value,
    epsilon=epsilon_value,
    cutoff=cutoff_value,
    offset=pmb.units.Quantity(0.0, 'nm')
)

pmb.define_particle(
    name='P1',
    z=0,
    sigma=sigma_value,
    epsilon=epsilon_value,
    cutoff=cutoff_value,
    offset=pmb.units.Quantity(0.0, 'nm')
)

# Define 'tNH' and 'pNH' if they will be used
pmb.define_particle(
    name='tNH',
    z=1,
    sigma=sigma_value,
    epsilon=epsilon_value,
    cutoff=cutoff_value,
    offset=pmb.units.Quantity(0.0, 'nm')
)

pmb.define_particle(
    name='pNH',
    z=1,
    sigma=sigma_value,
    epsilon=epsilon_value,
    cutoff=cutoff_value,
    offset=pmb.units.Quantity(0.0, 'nm')
)

# **2. Define Residues**
pmb.define_residue(name='Res0', central_bead='P0', side_chains=[])
pmb.define_residue(name='Res1', central_bead='P1', side_chains=[])

# **3. Define Molecules**
pmb.define_molecule(name='central_chain_molecule', residue_list=['Res0', 'Res0'])
pmb.define_molecule(name='branch_molecule', residue_list=['Res1'])

# **4. Define Bonds between Particles using define_bond**
bond_parameters = {
    'k': pmb.units.Quantity(1000.0, 'kJ/(nm**2)'),  # Adjust as needed
    'r_0': pmb.units.Quantity(1.0, 'nm')
}

# Bond between inner particles (central chain)
particle_pairs = [('P0', 'P0')]
pmb.define_bond(
    bond_type='harmonic',
    bond_parameters=bond_parameters,
    particle_pairs=particle_pairs
)

# Bond between inner and outer particles (branches)
particle_pairs = [('P0', 'P1')]
pmb.define_bond(
    bond_type='harmonic',
    bond_parameters=bond_parameters,
    particle_pairs=particle_pairs
)

# Bond between outer particles (if needed)
particle_pairs = [('P1', 'P1')]
pmb.define_bond(
    bond_type='harmonic',
    bond_parameters=bond_parameters,
    particle_pairs=particle_pairs
)

# **5. Initialize the Dendrimer Builder and Define Dendrimer**
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

system = espressomd.System(box_l=[20, 20, 20])

# Map particle names to type IDs
type_map = {'P0': 0, 'P1': 1, 'tNH': 2, 'pNH': 3}

# Map charges for functionalization
charge_map = {'tNH': 1.0, 'pNH': 1.0}

# **7. Create the dendrimer in the EspressoMD system**
dendrimer.create_dendrimer(system, pmb)

