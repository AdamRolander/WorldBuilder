// Minimal editor module so the plugin packages as a Fab "code plugin".
// All behaviour currently lives in Content/Python/worldbuilder_bridge.py;
// this module only registers the plugin and exposes a menu entry that runs it.
using UnrealBuildTool;

public class WorldBuilderBridge : ModuleRules
{
    public WorldBuilderBridge(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core" });

        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "CoreUObject", "Engine", "Slate", "SlateCore",
            "UnrealEd", "ToolMenus", "PythonScriptPlugin", "Projects"
        });
    }
}
