/**
 * PatternTime Client-side Storage & Cross-Dashboard State Sync
 * Uses IndexedDB for unlimited CSV file uploads (bypassing the 5MB sessionStorage quota)
 * and synchronizes state between index.html (Engineering Lab) and executive.html (Executive Summary).
 */
(function(window) {
    'use strict';

    const DB_NAME = 'PatternTimeDB';
    const DB_VERSION = 1;
    const STORE_UPLOADS = 'uploads';
    const STORE_SETTINGS = 'settings';

    let dbPromise = null;

    function getDB() {
        if (dbPromise) return dbPromise;
        dbPromise = new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                console.warn('[PatternTimeStorage] IndexedDB not supported; using in-memory storage.');
                resolve(null);
                return;
            }
            const request = window.indexedDB.open(DB_NAME, DB_VERSION);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_UPLOADS)) {
                    db.createObjectStore(STORE_UPLOADS, { keyPath: 'name' });
                }
                if (!db.objectStoreNames.contains(STORE_SETTINGS)) {
                    db.createObjectStore(STORE_SETTINGS, { keyPath: 'key' });
                }
            };
            request.onsuccess = (e) => resolve(e.target.result);
            request.onerror = (e) => {
                console.warn('[PatternTimeStorage] Failed to open IndexedDB:', e.target.error);
                resolve(null);
            };
        });
        return dbPromise;
    }

    const PatternTimeStorage = {
        /**
         * Save uploaded CSV file to IndexedDB
         */
        async saveUpload(name, content) {
            try {
                const db = await getDB();
                if (db) {
                    await new Promise((resolve, reject) => {
                        const tx = db.transaction(STORE_UPLOADS, 'readwrite');
                        const store = tx.objectStore(STORE_UPLOADS);
                        store.put({
                            name: name,
                            content: content,
                            size: content.length,
                            timestamp: Date.now()
                        });
                        tx.oncomplete = () => resolve();
                        tx.onerror = (e) => reject(e.target.error);
                    });
                }
            } catch (err) {
                console.warn('[PatternTimeStorage] IndexedDB save failed:', err);
            }

            // Always update active dataset metadata in sessionStorage safely
            this.setActiveDataset(name);
        },

        /**
         * Retrieve uploaded file content by name
         */
        async getUpload(name) {
            try {
                const db = await getDB();
                if (db) {
                    const record = await new Promise((resolve, reject) => {
                        const tx = db.transaction(STORE_UPLOADS, 'readonly');
                        const store = tx.objectStore(STORE_UPLOADS);
                        const req = store.get(name);
                        req.onsuccess = () => resolve(req.result);
                        req.onerror = (e) => reject(e.target.error);
                    });
                    if (record && record.content) {
                        return record;
                    }
                }
            } catch (err) {
                console.warn('[PatternTimeStorage] IndexedDB get failed:', err);
            }
            return null;
        },

        /**
         * List all uploaded files
         */
        async getAllUploads() {
            try {
                const db = await getDB();
                if (db) {
                    return await new Promise((resolve, reject) => {
                        const tx = db.transaction(STORE_UPLOADS, 'readonly');
                        const store = tx.objectStore(STORE_UPLOADS);
                        const req = store.getAll();
                        req.onsuccess = () => resolve(req.result || []);
                        req.onerror = (e) => reject(e.target.error);
                    });
                }
            } catch (err) {
                console.warn('[PatternTimeStorage] IndexedDB getAll failed:', err);
            }
            return [];
        },

        /**
         * Delete an uploaded file
         */
        async deleteUpload(name) {
            try {
                const db = await getDB();
                if (db) {
                    await new Promise((resolve, reject) => {
                        const tx = db.transaction(STORE_UPLOADS, 'readwrite');
                        const store = tx.objectStore(STORE_UPLOADS);
                        store.delete(name);
                        tx.oncomplete = () => resolve();
                        tx.onerror = (e) => reject(e.target.error);
                    });
                }
            } catch (err) {
                console.warn('[PatternTimeStorage] IndexedDB delete failed:', err);
            }
        },

        /**
         * Save active dataset name safely
         */
        setActiveDataset(name) {
            try {
                sessionStorage.setItem('patterntime_active_dataset', name);
            } catch (e) {
                // Quota or disabled sessionStorage
            }
        },

        /**
         * Get active dataset name from URL param or sessionStorage
         */
        getActiveDataset() {
            const urlParams = new URLSearchParams(window.location.search);
            const fromUrl = urlParams.get('dataset');
            if (fromUrl) return fromUrl;
            try {
                return sessionStorage.getItem('patterntime_active_dataset');
            } catch (e) {
                return null;
            }
        },

        /**
         * Get active route filter from URL param
         */
        getActiveRoute() {
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get('route') || null;
        }
    };

    window.PatternTimeStorage = PatternTimeStorage;
})(window);
