import hashlib

print(hashlib.new('md5', b'123').hexdigest())
print(hashlib.new('md5', b'3435').hexdigest())