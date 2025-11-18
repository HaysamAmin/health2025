import logging

# Setup del logger
logger = logging.getLogger("SymptomSphere")
logger.setLevel(logging.INFO)

# format to use in handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Handler to file
file_handler = logging.FileHandler("SymptomSphere.log")
file_handler.setFormatter(formatter)

# Handler to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
