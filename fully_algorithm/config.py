### Data path

DATA_PATH = '../sample_data'

### Features to extract from files in process_data

# Whether to group the rare materials into "Other" group (only preserving TOP 20 materials)
GROUP_RARE_MATERIAL = True

# Whether to change default material to non-default appearance (if applicable)
MATERIAL_TO_APPEARANCE = True

# Whether using TechNet embeddings (there needs to be TechNet embeddings in dataset)
TECHNET_EMBEDDING = True

# Whether using visual embeddings (there needs to be visual embeddings in dataset)
VISUAL_EMBEDDING = True

# Whether to consider link: "Shared Occurrence" - default is "FALSE"
SHARED_OCCURRENCE = False

# Whether to consider global features (features per graph/assembly)
GLOBAL_FEATURES = True

####################################################################################################

# TOP K dominant materials
DOMINANT_NUM = 20

# Placeholder global variable for setting ablated feature combinations
ABLATED_FEATURES = None

# TechNet embedding vector for keyword / body = "root"
ROOT_TECHNET_EMBEDDING = [0 for i in range(600)]
