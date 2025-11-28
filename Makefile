BUILD_DIR = build

all: $(BUILD_DIR)/Makefile
	$(MAKE) -C $(BUILD_DIR)

$(BUILD_DIR)/Makefile: CMakeLists.txt
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean