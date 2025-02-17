from Foundation import NSBundle
import objc

bundle_path = 'System/Library/Frameworks/CoreWLAN.framework'
objc.loadBundle('CoreWLAN',
                bundle_path = bundle_path,
                module_globals = globals())