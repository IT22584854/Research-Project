FROM node:22.14-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 8080

CMD ["sh", "-c", "printf \"window.__APP_CONFIG__ = { VITE_API_BASE: '%s' };\\n\" \"${VITE_API_BASE:-}\" > dist/runtime-config.js && npx serve -s dist -l tcp://0.0.0.0:${PORT:-8080} --no-clipboard"]