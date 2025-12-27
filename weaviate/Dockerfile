FROM node:20-alpine

WORKDIR app

# Copy package.json first and install deps
COPY package.json package-lock.json .
RUN npm install --production

# Copy the app code
COPY manage.js .

# Expose the port your app listens on
EXPOSE 9090

# Default command
CMD ["node", "manage.js", "http://weaviate:8080"]
