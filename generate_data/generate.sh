#!/bin/bash

# --- Configuration ---
PYTHON_SCRIPT="scene_gen2.py"
BASE_OUTPUT_DIR="final_dataset"
SCENES_PER_THEME=50
THEMES=("kitchen" "living_room" "office")
START_ID=0

# --- Main Execution ---
echo "🚀 Starting large-scale dataset generation..."
echo "Python Script:     $PYTHON_SCRIPT"
echo "Output Directory:    $BASE_OUTPUT_DIR"
echo "Themes:              ${THEMES[@]}"
echo "Scenes per Theme:    $SCENES_PER_THEME"
echo "------------------------------------------"

# Create the base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Initialize a global scene counter
global_scene_id=$START_ID

# Loop over each theme
for theme in "${THEMES[@]}"; do
    echo "Processing theme: $theme"
    
    # Create the theme-specific directory
    THEME_DIR="$BASE_OUTPUT_DIR/$theme"
    mkdir -p "$THEME_DIR"

    # Loop to generate scenes for the current theme
    for i in $(seq 1 $SCENES_PER_THEME); do
        
        # Format the scene ID with leading zeros
        SCENE_ID_FORMATTED=$(printf "%04d" $global_scene_id)
        SCENE_OUTPUT_DIR="$THEME_DIR/scene_$SCENE_ID_FORMATTED"

        echo "-> Generating Scene ID: $SCENE_ID_FORMATTED (Theme: $theme, Run: $i/$SCENES_PER_THEME)"
        
        # Run the BlenderProc script with required arguments:
        # blenderproc run <script> -- <scene_id> <output_dir> <theme>
        blenderproc run "$PYTHON_SCRIPT" -- "$global_scene_id" "$SCENE_OUTPUT_DIR" "$theme"

        # Check if the last command was successful
        if [ $? -eq 0 ]; then
            echo "   ✓ Scene $SCENE_ID_FORMATTED completed successfully."
        else
            echo "   ✗ Scene $SCENE_ID_FORMATTED failed. Check logs for details."
            # Optional: To stop the entire process on a single failure, uncomment the next line
            # exit 1
        fi
        
        # Increment the global scene ID for the next run
        global_scene_id=$((global_scene_id + 1))
    done
    echo "------------------------------------------"
done

echo "🎉 All scenes generated!"
echo "Total Scenes: $((global_scene_id - START_ID))"
echo "Dataset located in: $BASE_OUTPUT_DIR"