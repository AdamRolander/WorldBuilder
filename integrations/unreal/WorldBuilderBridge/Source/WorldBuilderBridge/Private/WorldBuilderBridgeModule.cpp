// WorldBuilder Bridge — editor module.
// Adds Tools > WorldBuilder > "Generate scene from photo…" which opens a file
// dialog and hands the path to the Python bridge. Everything else is Python.
#include "Modules/ModuleManager.h"
#include "ToolMenus.h"
#include "IPythonScriptPlugin.h"
#include "DesktopPlatformModule.h"
#include "IDesktopPlatform.h"
#include "Misc/Paths.h"
#include "Interfaces/IPluginManager.h"
#include "Framework/Application/SlateApplication.h"

#define LOCTEXT_NAMESPACE "WorldBuilderBridge"

class FWorldBuilderBridgeModule : public IModuleInterface
{
public:
    virtual void StartupModule() override
    {
        UToolMenus::RegisterStartupCallback(FSimpleMulticastDelegate::FDelegate::CreateRaw(
            this, &FWorldBuilderBridgeModule::RegisterMenus));
    }

    virtual void ShutdownModule() override
    {
        UToolMenus::UnRegisterStartupCallback(this);
        UToolMenus::UnregisterOwner(this);
    }

private:
    void RegisterMenus()
    {
        FToolMenuOwnerScoped OwnerScoped(this);
        UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Tools");
        FToolMenuSection& Section = Menu->FindOrAddSection("WorldBuilder");
        Section.Label = LOCTEXT("WorldBuilderSection", "WorldBuilder");
        Section.AddMenuEntry(
            "WorldBuilderGenerate",
            LOCTEXT("Generate", "Generate scene from photo..."),
            LOCTEXT("GenerateTooltip", "Upload a photo to your WorldBuilder server and import the 3D scene"),
            FSlateIcon(),
            FUIAction(FExecuteAction::CreateRaw(this, &FWorldBuilderBridgeModule::OnGenerate)));
    }

    void OnGenerate()
    {
        IDesktopPlatform* Desktop = FDesktopPlatformModule::Get();
        if (!Desktop) return;
        TArray<FString> Files;
        const void* Parent = FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
        if (!Desktop->OpenFileDialog(Parent, TEXT("Choose a room photo"), FPaths::ProjectDir(), TEXT(""),
                                     TEXT("Images (*.jpg;*.jpeg;*.png;*.webp)|*.jpg;*.jpeg;*.png;*.webp"),
                                     EFileDialogFlags::None, Files) || Files.Num() == 0)
        {
            return;
        }
        // Make the plugin's Content/Python importable, then call the bridge.
        const FString PyDir = FPaths::Combine(
            IPluginManager::Get().FindPlugin(TEXT("WorldBuilderBridge"))->GetContentDir(), TEXT("Python"));
        const FString Photo = Files[0].Replace(TEXT("\\"), TEXT("/"));
        const FString Cmd = FString::Printf(
            TEXT("import sys; sys.path.append(r'%s'); import importlib, worldbuilder_bridge as wb; "
                 "importlib.reload(wb); wb.generate_and_import(r'%s')"), *PyDir, *Photo);
        if (IPythonScriptPlugin::Get()->IsPythonAvailable())
        {
            IPythonScriptPlugin::Get()->ExecPythonCommand(*Cmd);
        }
    }
};

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FWorldBuilderBridgeModule, WorldBuilderBridge)
