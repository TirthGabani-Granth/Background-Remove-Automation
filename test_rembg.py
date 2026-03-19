import inspect
import rembg

print("rembg.remove signature:")
print(inspect.signature(rembg.remove))

print("Sessions available:")
# We can try to list models from rembg
try:
    from rembg.session_factory import new_session
    print("new_session args:")
    print(inspect.signature(new_session))
except Exception as e:
    print(e)
