import type { Activity, Conversation, ConversationSession } from './types';

const DB_NAME = 'local-tools-lab.memory.v1';
const DB_VERSION = 1;
const ACTIVE_CONVERSATION_KEY = 'active-conversation';
export const MAX_EVENTS_PER_CONVERSATION = 100;

type MemorySnapshot = {
  activeConversation: Conversation;
  conversations: Conversation[];
  activity: Activity[];
  session: ConversationSession;
};

function requestResult<T>(request: IDBRequest<T>) {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'));
  });
}

function transactionDone(transaction: IDBTransaction) {
  return new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed.'));
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB transaction was aborted.'));
  });
}

let memoryDb: Promise<IDBDatabase> | null = null;

/** One shared connection; reopened only if the browser or another tab closes it. */
function openMemoryDb() {
  memoryDb ??= new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      const conversations = db.createObjectStore('conversations', { keyPath: 'id' });
      conversations.createIndex('updatedAt', 'updatedAt');
      const events = db.createObjectStore('events', { keyPath: 'id' });
      events.createIndex('conversationId', 'conversationId');
      db.createObjectStore('sessions', { keyPath: 'conversationId' });
      db.createObjectStore('settings', { keyPath: 'key' });
    };
    request.onsuccess = () => {
      const db = request.result;
      // Let another tab upgrade the schema instead of blocking it.
      db.onversionchange = () => {
        db.close();
        memoryDb = null;
      };
      db.onclose = () => { memoryDb = null; };
      resolve(db);
    };
    request.onerror = () => {
      memoryDb = null;
      reject(request.error ?? new Error('Could not open conversation memory.'));
    };
  });
  return memoryDb;
}

/** Test hook: drop the cached connection so a fresh database can be opened. */
export function closeMemoryDb() {
  memoryDb?.then((db) => db.close()).catch(() => undefined);
  memoryDb = null;
}

function newConversation(title = 'New conversation'): Conversation {
  const now = new Date().toISOString();
  return { id: crypto.randomUUID(), title, createdAt: now, updatedAt: now };
}

const emptySession = (conversationId: string): ConversationSession => ({
  conversationId,
  planReview: null,
  pendingClarification: null,
  refiningExecutedPlan: false
});

const newestFirst = (left: Activity, right: Activity) =>
  (right.order ?? (Date.parse(right.createdAt ?? '') || 0)) - (left.order ?? (Date.parse(left.createdAt ?? '') || 0));

export function deduplicateActivity(activity: Activity[]) {
  const seen = new Set<string>();
  return activity.filter((item) => {
    // Earlier development builds could invoke a React state updater twice in
    // Strict Mode. Those copies share the exact source, message, and timestamp.
    const key = `${item.source}\u0000${item.message}\u0000${item.createdAt ?? item.at}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function loadMemory(): Promise<MemorySnapshot> {
  const db = await openMemoryDb();
  // A single write transaction makes first-run initialization safe when
  // React Strict Mode mounts the app twice during development.
  const read = db.transaction(['conversations', 'settings'], 'readwrite');
  const readDone = transactionDone(read);
  const conversationsRequest = read.objectStore('conversations').getAll();
  const activeRequest = read.objectStore('settings').get(ACTIVE_CONVERSATION_KEY);
  const [existing, activeSetting] = await Promise.all([
    requestResult(conversationsRequest) as Promise<Conversation[]>,
    requestResult(activeRequest) as Promise<{ key: string; value: string } | undefined>
  ]);

  let conversations = existing;
  let activeConversation = conversations.find((item) => item.id === activeSetting?.value);
  if (!activeConversation) {
    activeConversation = conversations.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))[0] ?? newConversation();
    read.objectStore('conversations').put(activeConversation);
    read.objectStore('settings').put({ key: ACTIVE_CONVERSATION_KEY, value: activeConversation.id });
    conversations = [...conversations.filter((item) => item.id !== activeConversation!.id), activeConversation];
  }
  await readDone;

  const detail = db.transaction(['events', 'sessions'], 'readonly');
  const detailDone = transactionDone(detail);
  const activityRequest = detail.objectStore('events').index('conversationId').getAll(activeConversation.id);
  const sessionRequest = detail.objectStore('sessions').get(activeConversation.id);
  const [activity, session] = await Promise.all([
    requestResult(activityRequest) as Promise<Activity[]>,
    requestResult(sessionRequest) as Promise<ConversationSession | undefined>
  ]);
  await detailDone;
  return {
    activeConversation,
    conversations: conversations.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt)),
    activity: deduplicateActivity(activity.sort(newestFirst)),
    session: session ?? emptySession(activeConversation.id)
  };
}

export async function appendMemoryEvent(conversationId: string, event: Activity) {
  const db = await openMemoryDb();
  const transaction = db.transaction(['events', 'conversations'], 'readwrite');
  const done = transactionDone(transaction);
  const events = transaction.objectStore('events');
  events.put({ ...event, conversationId });
  const conversations = transaction.objectStore('conversations');
  const current = await requestResult(conversations.get(conversationId)) as Conversation | undefined;
  if (current) {
    const isFirstUserMessage = event.source === 'person' && current.title === 'New conversation';
    conversations.put({
      ...current,
      title: isFirstUserMessage ? event.message.replace(/\s+/g, ' ').trim().slice(0, 54) : current.title,
      updatedAt: event.createdAt ?? new Date().toISOString()
    });
  }
  // Keep only the newest events per conversation so stored history stays bounded.
  const byConversation = events.index('conversationId');
  const count = await requestResult(byConversation.count(conversationId));
  if (count > MAX_EVENTS_PER_CONVERSATION) {
    const stored = await requestResult(byConversation.getAll(conversationId)) as Activity[];
    stored.sort(newestFirst).slice(MAX_EVENTS_PER_CONVERSATION).forEach((item) => events.delete(item.id));
  }
  await done;
}

export async function saveConversationSession(session: ConversationSession) {
  const db = await openMemoryDb();
  const transaction = db.transaction('sessions', 'readwrite');
  const done = transactionDone(transaction);
  transaction.objectStore('sessions').put(session);
  await done;
}

export async function createMemoryConversation() {
  const conversation = newConversation();
  const db = await openMemoryDb();
  const transaction = db.transaction(['conversations', 'settings', 'sessions'], 'readwrite');
  const done = transactionDone(transaction);
  transaction.objectStore('conversations').put(conversation);
  transaction.objectStore('settings').put({ key: ACTIVE_CONVERSATION_KEY, value: conversation.id });
  transaction.objectStore('sessions').put(emptySession(conversation.id));
  await done;
  return conversation;
}

export async function selectMemoryConversation(conversationId: string) {
  const db = await openMemoryDb();
  const transaction = db.transaction(['settings', 'events', 'sessions'], 'readwrite');
  const done = transactionDone(transaction);
  transaction.objectStore('settings').put({ key: ACTIVE_CONVERSATION_KEY, value: conversationId });
  const activityRequest = transaction.objectStore('events').index('conversationId').getAll(conversationId);
  const sessionRequest = transaction.objectStore('sessions').get(conversationId);
  const [activity, session] = await Promise.all([
    requestResult(activityRequest) as Promise<Activity[]>,
    requestResult(sessionRequest) as Promise<ConversationSession | undefined>
  ]);
  await done;
  return {
    activity: deduplicateActivity(activity.sort(newestFirst)),
    session: session ?? emptySession(conversationId)
  };
}

export async function clearMemory() {
  const db = await openMemoryDb();
  const conversation = newConversation();
  const transaction = db.transaction(['conversations', 'events', 'sessions', 'settings'], 'readwrite');
  const done = transactionDone(transaction);
  transaction.objectStore('conversations').clear();
  transaction.objectStore('events').clear();
  transaction.objectStore('sessions').clear();
  transaction.objectStore('settings').clear();
  transaction.objectStore('conversations').put(conversation);
  transaction.objectStore('sessions').put(emptySession(conversation.id));
  transaction.objectStore('settings').put({ key: ACTIVE_CONVERSATION_KEY, value: conversation.id });
  await done;
  return conversation;
}
