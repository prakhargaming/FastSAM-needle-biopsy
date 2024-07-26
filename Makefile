# Define the output directory
OUTPUT_DIR := output

# Target to clean the output directory
clean:
	@echo "Cleaning output directory..."
	@find $(OUTPUT_DIR) -type f -delete
	@echo "Output directory cleaned."