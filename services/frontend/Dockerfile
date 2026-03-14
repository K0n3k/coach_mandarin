# --- Stage 1 : build ---
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

# --- Stage 2 : serve ---
FROM nginx:alpine

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

COPY --from=build /app/build /usr/share/nginx/html

# nginx.conf is mounted as a volume from docker-compose.yml
# Provide a minimal fallback in case it's run standalone
RUN printf 'server { listen 8080; root /usr/share/nginx/html; location / { try_files $uri $uri/ /index.html; } }\n' \
    > /etc/nginx/conf.d/default.conf

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
