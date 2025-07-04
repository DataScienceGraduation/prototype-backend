import jwt
from django.conf import settings
from rest_framework import authentication, exceptions
from .models import User, APIKey

JWT_ALGORITHM = 'HS256'

class JWTAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization')

        if not auth_header or not auth_header.startswith('Bearer '):
            return None

        try:
            token = auth_header.split(' ')[1]
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            username = payload['username']
            user = User.objects.get(username=username)
            
            # Attach payload to request for compatibility
            request.jwt_payload = payload
            
            return (user, token)

        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed('Token expired')
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed('Invalid token')
        except User.DoesNotExist:
            raise exceptions.AuthenticationFailed('User not found')
        except Exception as e:
            raise exceptions.AuthenticationFailed(f'Authentication error: {e}')

class APIKeyAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None

        key = auth_header.split(' ')[1]
        try:
            api_key_obj = APIKey.objects.get(key=key)
            return (api_key_obj.user, None)
        except APIKey.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid API Key')
