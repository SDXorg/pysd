# -*- mode: python -*-

block_cipher = None


a = Analysis(['wx_xmile2r.py'],
             pathex=['C:\\Users\\Miguel\\Documents\\0 Versiones\\2 Proyectos\\pysd\\translator_xmile'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='wx_xmile2r',
          debug=False,
          strip=False,
          upx=True,
          console=True )
